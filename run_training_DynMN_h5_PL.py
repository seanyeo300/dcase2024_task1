import wandb
import numpy as np
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import argparse
from sklearn import metrics
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import torchaudio
import json

from dataset.dcase24_ntu_DynMN import ntu_get_training_set_dir, ntu_get_test_set, ntu_get_eval_set, open_h5, close_h5, dataset_config
# from models.mn.model import get_model as get_mobilenet
from models.dymn.model import get_model as get_dymn
from models.preprocess import AugmentMelSTFT
from helpers.init import worker_init_fn
from helpers.utils import NAME_TO_WIDTH, exp_warmup_linear_down, mixup, mixstyle
from helpers import nessi

def load_and_modify_checkpoint(pl_module,num_classes=10):
    print("Write modify ckpt script for DyMN")
    pass # not implemented yet
    # Modify the final layer
    # pl_module.model.head = nn.Sequential(
    #     nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True),
    #     nn.Linear(768, num_classes)
    # )
    # pl_module.model.head_dist = nn.Linear(768, num_classes)
    return pl_module

class PLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # model to preprocess waveforms into log mel spectrograms
        # model to preprocess waveform into mel spectrograms
        self.mel = AugmentMelSTFT(n_mels=config.n_mels,
                         sr=config.resample_rate,
                         win_length=config.window_size,
                         hopsize=config.hop_size,
                         n_fft=config.n_fft,
                         freqm=config.freqm,
                         timem=config.timem,
                         fmin=config.fmin,
                         fmax=config.fmax,
                         fmin_aug_range=config.fmin_aug_range,
                         fmax_aug_range=config.fmax_aug_range
                         )
        
        # logic for loading DyMN
        self.model_name = config.model_name
        self.pretrained_name = self.model_name if config.pretrained else None
        self.width = NAME_TO_WIDTH(self.model_name) if self.model_name and config.pretrained else config.model_width
        self.model = get_dymn(width_mult=self.width, pretrained_name=self.pretrained_name,
                         pretrain_final_temp=config.pretrain_final_temp,
                         num_classes=config.num_classes)
        
        self.device_ids = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']
        self.label_ids = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall',
                          'street_pedestrian', 'street_traffic', 'tram']
        self.device_groups = {'a': "real", 'b': "real", 'c': "real",
                              's1': "seen", 's2': "seen", 's3': "seen",
                              's4': "unseen", 's5': "unseen", 's6': "unseen"}
        self.calc_device_info = True
        self.epoch = 0
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
    def mel_forward(self, x):
        old_shape = x.size()
        x = x.reshape(-1, old_shape[2])
        x = self.mel(x)
        x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
        return x
    
    def forward(self, x):
        return self.model(x)
    
    def predict_step(self, eval_batch, batch_idx, dataloader_idx=0):
        x, files = eval_batch
        x = self.mel_forward(x)
        # x = x.half()
        y_hat, embed = self.forward(x)
        return files,y_hat
    
    def training_step(self, batch, batch_idx):
    
        x, files, y, dev, city, index , logits = batch
        bs = x.size(0)
        y=y.long()
        x = self.mel_forward(x)
        if self.config.mixstyle_p > 0:   
            x = mixstyle(x, self.config.mixstyle_p, self.config.mixstyle_alpha)
            y_hat, _ = self.forward(x)
            samples_loss = F.cross_entropy(y_hat, y, reduction="none")
        elif self.config.mixup_alpha:
            rn_indices, lam = mixup(bs, self.config.mixup_alpha)
            lam = lam.to(x.device)
            x = x * lam.reshape(bs, 1, 1, 1) + \
                x[rn_indices] * (1. - lam.reshape(bs, 1, 1, 1))
            y_hat, _ = self.forward(x)
            samples_loss = (F.cross_entropy(y_hat, y, reduction="none") * lam.reshape(bs) +
                            F.cross_entropy(y_hat, y[rn_indices], reduction="none") * (
                                        1. - lam.reshape(bs)))

        else:
            y_hat, _ = self.forward(x)
            samples_loss = F.cross_entropy(y_hat, y, reduction="none")

        # loss
        loss = samples_loss.mean()
        
        _, preds = torch.max(y_hat, dim=1)
        n_correct_pred = (preds == y).sum()
        results = {"loss": loss, "n_correct_pred": n_correct_pred, "n_pred": len(y)}
        if self.calc_device_info:
            devices = [d.rsplit("-", 1)[1][:-4] for d in files]

            for d in self.device_ids:
                results["devloss." + d] = torch.as_tensor(0., device=self.device)
                results["devcnt." + d] = torch.as_tensor(0., device=self.device)

            for i, d in enumerate(devices):
                results["devloss." + d] = results["devloss." + d] + samples_loss[i]
                results["devcnt." + d] = results["devcnt." + d] + 1.

        return results
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        train_acc = sum([x['n_correct_pred'] for x in outputs]) * 1.0 / sum(x['n_pred'] for x in outputs)
        logs = {'train.loss': avg_loss, 'train_acc': train_acc}
        if self.calc_device_info:
            for d in self.device_ids:
                dev_loss = torch.stack([x["devloss." + d] for x in outputs]).sum()
                dev_cnt = torch.stack([x["devcnt." + d] for x in outputs]).sum()
                logs["tloss." + d] = dev_loss / dev_cnt
                logs["tcnt." + d] = dev_cnt

        self.log_dict(logs)

        print(f"Training Loss: {avg_loss}")
        print(f"Training Accuracy: {train_acc}")
        
    def validation_step(self, batch, batch_idx):
        x, f, y, dev, city, index = batch

        x = self.mel_forward(x)

        y_hat, embed = self.forward(x)
        y = y.long()
        samples_loss = F.cross_entropy(y_hat, y, reduction="none")
        loss = samples_loss.mean()

        self.log("validation.loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        _, preds = torch.max(y_hat, dim=1)
        n_correct_pred_per_sample = (preds == y)
        n_correct_pred = n_correct_pred_per_sample.sum()
        results = {"val_loss": loss, "n_correct_pred": n_correct_pred, "n_pred": len(y)}
        return results
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        val_acc = sum([x['n_correct_pred'] for x in outputs]) * 1.0 / sum(x['n_pred'] for x in outputs)
        logs = {'val.loss': avg_loss, 'val_acc': val_acc}
        self.log_dict(logs)
        if self.epoch > 0:
            print()
            print(f"Validation Loss: {avg_loss}")
            print(f"Validation Accuracy: {val_acc}")

        self.epoch += 1
        
    def test_step(self, test_batch, batch_idx):
        x, files, labels, devices, cities, index = test_batch
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device=x.device)
        # maximum memory allowance for parameters: 128 KB
        # baseline has 61148 parameters -> we can afford 16-bit precision
        # since 61148 * 16 bit ~ 122 kB
           
        x = self.mel_forward(x)

        y_hat, embed = self.forward(x)
        labels = labels.long()
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")
        # loss = samples_loss.mean()

        # for computing accuracy
        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()

        dev_names = [d.rsplit("-", 1)[1][:-4] for d in files]
        results = {'loss': samples_loss.mean(), "n_correct": n_correct,
                   "n_pred": torch.as_tensor(len(labels), device=self.device)}

        # log metric per device and scene
        for d in self.device_ids:
            results["devloss." + d] = torch.as_tensor(0., device=self.device)
            results["devcnt." + d] = torch.as_tensor(0., device=self.device)
            results["devn_correct." + d] = torch.as_tensor(0., device=self.device)
        for i, d in enumerate(dev_names):
            results["devloss." + d] = results["devloss." + d] + samples_loss[i]
            results["devn_correct." + d] = results["devn_correct." + d] + n_correct_per_sample[i]
            results["devcnt." + d] = results["devcnt." + d] + 1

        for l in self.label_ids:
            results["lblloss." + l] = torch.as_tensor(0., device=self.device)
            results["lblcnt." + l] = torch.as_tensor(0., device=self.device)
            results["lbln_correct." + l] = torch.as_tensor(0., device=self.device)
        for i, l in enumerate(labels):
            results["lblloss." + self.label_ids[l]] = results["lblloss." + self.label_ids[l]] + samples_loss[i]
            results["lbln_correct." + self.label_ids[l]] = \
                results["lbln_correct." + self.label_ids[l]] + n_correct_per_sample[i]
            results["lblcnt." + self.label_ids[l]] = results["lblcnt." + self.label_ids[l]] + 1
        self.test_step_outputs.append(results)
        
    def on_test_epoch_end(self):
        # convert a list of dicts to a flattened dict
        outputs = {k: [] for k in self.test_step_outputs[0]}
        for step_output in self.test_step_outputs:
            for k in step_output:
                outputs[k].append(step_output[k])
        for k in outputs:
            outputs[k] = torch.stack(outputs[k])

        avg_loss = outputs['loss'].mean()
        acc = sum(outputs['n_correct']) * 1.0 / sum(outputs['n_pred'])

        logs = {'acc': acc, 'loss': avg_loss}

        # log metric per device and scene
        for d in self.device_ids:
            dev_loss = outputs["devloss." + d].sum()
            dev_cnt = outputs["devcnt." + d].sum()
            dev_corrct = outputs["devn_correct." + d].sum()
            logs["loss." + d] = dev_loss / dev_cnt
            logs["acc." + d] = dev_corrct / dev_cnt
            logs["cnt." + d] = dev_cnt
            # device groups
            logs["acc." + self.device_groups[d]] = logs.get("acc." + self.device_groups[d], 0.) + dev_corrct
            logs["count." + self.device_groups[d]] = logs.get("count." + self.device_groups[d], 0.) + dev_cnt
            logs["lloss." + self.device_groups[d]] = logs.get("lloss." + self.device_groups[d], 0.) + dev_loss

        for d in set(self.device_groups.values()):
            logs["acc." + d] = logs["acc." + d] / logs["count." + d]
            logs["lloss." + d] = logs["lloss." + d] / logs["count." + d]

        for l in self.label_ids:
            lbl_loss = outputs["lblloss." + l].sum()
            lbl_cnt = outputs["lblcnt." + l].sum()
            lbl_corrct = outputs["lbln_correct." + l].sum()
            logs["loss." + l] = lbl_loss / lbl_cnt
            logs["acc." + l] = lbl_corrct / lbl_cnt
            logs["cnt." + l] = lbl_cnt

        logs["macro_avg_acc"] = torch.mean(torch.stack([logs["acc." + l] for l in self.label_ids]))
        # prefix with 'test' for logging
        self.log_dict({"test/" + k: logs[k] for k in logs})
        self.test_step_outputs.clear()
    def configure_optimizers(self):
        """
        This is the way pytorch lightening requires optimizers and learning rate schedulers to be defined.
        The specified items are used automatically in the optimization loop (no need to call optimizer.step() yourself).
        :return: dict containing optimizer and learning rate scheduler
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        # phases of lr schedule: exponential increase, constant lr, linear decrease, fine-tune
        schedule_lambda = \
            exp_warmup_linear_down(self.config.warm_up_len, self.config.ramp_down_len, self.config.ramp_down_start,
                                   self.config.last_lr_value)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }

def train(config):
    # Train Models for Acoustic Scene Classification

   # logging is done using wandb
    wandb_logger = WandbLogger(
        project=config.project_name,
        notes="DyMN Models for Acoustic Scene Classification.",
        tags=["Tau Urban Acoustic Scenes 2022 Mobile", "Acoustic Scene Classification", "Fine-Tuning"],
        config=config,
        name=config.experiment_name
    )
    # train dataloader
    assert config.subset in {100, 50, 25, 10, 5,"cochl10s"}, "Specify an integer value in: {100, 50, 25, 10, 5} to use one of " \
                                                  "the given subsets."
    # get pointer to h5 file containing audio samples
    hf_in = open_h5('h5py_audio_wav')
    hmic_in = open_h5('h5py_mic_wav_1')
    # dataloader
    train_dl = DataLoader(dataset=ntu_get_training_set_dir(config.subset, config.dir_prob,
                                                    roll=False if config.no_roll else True,
                                                    wavmix=False if config.no_wavmix else True,
                                                    gain_augment=config.gain_augment, hf_in=hf_in, hmic_in=hmic_in),               
                          worker_init_fn=worker_init_fn,
                          num_workers=config.num_workers,
                          batch_size=config.batch_size,
                          pin_memory= True,
                          shuffle=True)
    # evaluation loader
    test_dl = DataLoader(dataset=ntu_get_test_set(hf_in),
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size,
                         pin_memory=True)
    # create pytorch lightening module
    ckpt_id = None if config.ckpt_id == "None" else config.ckpt_id
    if ckpt_id is not None:
        ckpt_dir = os.path.join(config.project_name, config.ckpt_id, "checkpoints")
        assert os.path.exists(ckpt_dir), f"No such folder: {ckpt_dir}"
        #ckpt_file = os.path.join(ckpt_dir, "last.ckpt")
        for file in os.listdir(ckpt_dir):
            if "epoch" in file:
                ckpt_file = os.path.join(ckpt_dir,file) # choosing the best model ckpt
                print(f"found ckpt file: {file}")
        pl_module = PLModule.load_from_checkpoint(ckpt_file, config=config)
    else:
        pl_module = PLModule(config) # this initializes the model pre-trained on audioset

    # name = None
    # accuracy, val_loss = float('NaN'), float('NaN')
    
    
    # get model complexity from nessi and log results to wandb
    sample = next(iter(train_dl))[0][0].unsqueeze(0)
    shape = pl_module.mel_forward(sample).size()
    macs, params = nessi.get_torch_size(pl_module.model, input_size=shape)
    wandb_logger.experiment.config['MACs'] = macs
    wandb_logger.experiment.config['Parameters'] = params
    
    # create monitor to keep track of learning rate - we want to check the behaviour of our learning rate schedule
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_last=True, 
        monitor="validation.loss", 
        save_top_k=1)
    trainer = pl.Trainer(max_epochs=config.n_epochs,
                         logger=wandb_logger,
                         accelerator='gpu',
                         devices=[1],
                         callbacks=[lr_monitor, checkpoint_callback])
    # start training and validation for the specified number of epochs
    trainer.fit(pl_module, train_dl, test_dl)
    trainer.test(ckpt_path='best', dataloaders=test_dl)
    ############ h5 edit end #################
    # close file pointer to h5 file 
    close_h5(hf_in)
    close_h5(hmic_in)
    
    wandb.finish()

def evaluate(config):
    import os
    from sklearn import preprocessing
    import pandas as pd
    import torch.nn.functional as F
    # from datasets.dcase23_dev import dataset_config

    assert config.ckpt_id is not None, "A value for argument 'ckpt_id' must be provided."
    ckpt_dir = os.path.join(config.project_name, config.ckpt_id, "checkpoints")
    assert os.path.exists(ckpt_dir), f"No such folder: {ckpt_dir}"
    #ckpt_file = os.path.join(ckpt_dir, "last.ckpt")
    for file in os.listdir(ckpt_dir):
        if "epoch" in file:
            ckpt_file = os.path.join(ckpt_dir,file) # choosing the best model ckpt
            print(f"found ckpt file: {file}")
    # ckpt_file = os.path.join(ckpt_dir, "xyz.ckpt") # Change the path to the model path desired
    assert os.path.exists(ckpt_file), f"No such file: {ckpt_file}. Implement your own mechanism to select" \
                                      f"the desired checkpoint."

    # create folder to store predictions
    os.makedirs("predictions", exist_ok=True)
    out_dir = os.path.join("predictions", config.ckpt_id)
    os.makedirs(out_dir, exist_ok=True)

    # load lightning module from checkpoint
    pl_module = PLModule.load_from_checkpoint(ckpt_file, config=config)
    
    ############# h5 edit here ##############
    # Open h5 file once
    hf_in = open_h5('h5py_audio_wav')
    eval_hf = open_h5('h5py_audio_wav2') # only when obtaining pre-computed train
    # eval_hf = open_h5('h5py_audio_eval_wav')
    # load lightning module from checkpoint
    pl_module = PLModule.load_from_checkpoint(ckpt_file, config=config)
    trainer = pl.Trainer(logger=False,
                         accelerator='gpu',
                         devices=1,
                         precision=config.precision)
    ############# h5 edit here ##############
    # evaluate lightning module on development-test split
    test_dl = DataLoader(dataset=ntu_get_test_set(hf_in),
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size,
                         pin_memory=True)

    # get model complexity from nessi
    sample = next(iter(test_dl))[0][0].unsqueeze(0).to(pl_module.device)
    shape = pl_module.mel_forward(sample).size()
    macs, params = nessi.get_torch_size(pl_module.model, input_size=shape)
    print(f"Model Complexity: MACs: {macs}, Params: {params}")
    
    # obtain and store details on model for reporting in the technical report
    info = {}
    info['MACs'] = macs
    info['Params'] = params
    res = trainer.test(pl_module, test_dl,ckpt_path=ckpt_file)
    info['test'] = res

    ############# h5 edit here ##############
    # generate predictions on evaluation set
    eval_dl = DataLoader(dataset=ntu_get_eval_set(hf_in),
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)
    
    predictions = trainer.predict(pl_module, dataloaders=eval_dl, ckpt_path=ckpt_file) # predictions returns as files, y_hat
    # all filenames
    all_files = [item[len("audio/"):] for files, _ in predictions for item in files]
    # all predictions
    logits = torch.cat([torch.as_tensor(p) for _, p in predictions], 0)
    all_predictions = F.softmax(logits.float(), dim=1)

    # write eval set predictions to csv file
    df = pd.read_csv(dataset_config['meta_csv'], sep="\t")
    le = preprocessing.LabelEncoder()
    le.fit_transform(df[['scene_label']].values.reshape(-1))
    class_names = le.classes_
    df = {'filename': all_files}
    scene_labels = [class_names[i] for i in torch.argmax(all_predictions, dim=1)]
    df['scene_label'] = scene_labels
    for i, label in enumerate(class_names):
        df[label] = logits[:, i]
    df = pd.DataFrame(df)

    # save eval set predictions, model state_dict and info to output folder
    df.to_csv(os.path.join(out_dir, 'output.csv'), sep='\t', index=False)
    torch.save(pl_module.model.state_dict(), os.path.join(out_dir, "model_state_dict.pt"))
    with open(os.path.join(out_dir, "info.json"), "w") as json_file:
        json.dump(info, json_file)
        
    ############# h5 edit here ##############
    close_h5(hf_in)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')

    # general
    parser.add_argument('--project_name', type=str, default="NTU_ASC24_DynMN")
    parser.add_argument('--experiment_name', type=str, default="tDynMN10_FTtau_32K_FMS_DIR_sub10_fixh5")
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--batch_size', type=int, default=48) # default = 32 ; JS = 48
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--cache_path', type=str, default=None)
    parser.add_argument('--num_classes',type=int,default=10)
    parser.add_argument('--subset', type=int, default=10)
    
    # evaluation
    parser.add_argument('--evaluate', action='store_true')  # predictions on eval set
    parser.add_argument('--ckpt_id', type=str, required=False, default=None)
    
    # training
    parser.add_argument('--pretrained', action='store_true', default=True) # Pre-trained on AS
    parser.add_argument('--model_name', type=str, default="dymn10_as") # Best MAP model
    parser.add_argument('--pretrain_final_temp', type=float, default=1.0)  # for DyMN
    parser.add_argument('--model_width', type=float, default=1.0)
    parser.add_argument('--head_type', type=str, default="mlp")
    parser.add_argument('--se_dims', type=str, default="c")
    parser.add_argument('--n_epochs', type=int, default=80) # default=80
    parser.add_argument('--mixup_alpha', type=float, default=0.3)
    parser.add_argument('--mixstyle_p', type=float, default=0.4)
    parser.add_argument('--mixstyle_alpha', type=float, default=0.4)
    parser.add_argument('--no_roll', action='store_true', default=False)
    parser.add_argument('--no_wavmix', action='store_true', default=True) #enforces no mixup
    parser.add_argument('--gain_augment', type=int, default=12)
    parser.add_argument('--weight_decay', type=int, default=0.0) #ADAM, no WD
    parser.add_argument('--dir_prob', type=float, default=0.6)  # prob. to apply device impulse response augmentation, default for TAU = 0.6 ; JS does not use it

    # lr schedule
    parser.add_argument('--lr', type=float, default=1e-4) # JS setting, TAU'19 = 0.003
    parser.add_argument('--warm_up_len', type=int, default=10)
    parser.add_argument('--ramp_down_start', type=int, default=10)
    parser.add_argument('--ramp_down_len', type=int, default=65)
    parser.add_argument('--last_lr_value', type=float, default=0.01)

    # preprocessing
    parser.add_argument('--orig_sample_rate', type = int, default = 44100)
    parser.add_argument('--resample_rate', type=int, default=32000) # JS does not use 44.1K
    parser.add_argument('--window_size', type=int, default=800)
    parser.add_argument('--hop_size', type=int, default=505) # default = 320 ; JS = 505
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--freqm', type=int, default=48) # default was 0, JS uses freqM
    parser.add_argument('--timem', type=int, default=0)
    parser.add_argument('--fmin', type=int, default=0)
    parser.add_argument('--fmax', type=int, default=None)
    parser.add_argument('--fmin_aug_range', type=int, default=10)
    parser.add_argument('--fmax_aug_range', type=int, default=2000)

    args = parser.parse_args()
    if args.evaluate:
        evaluate(args)
    else:
        train(args)
