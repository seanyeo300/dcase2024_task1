import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
import torchaudio
from torch.utils.data import DataLoader
import argparse
import torch.nn.functional as F
import transformers
import wandb
import json
import os
import torch.nn as nn
from torch.autograd import Variable
from dataset.dcase24_ntu_teacher_tv1 import ntu_get_training_set_dir, ntu_get_test_set, ntu_get_eval_set, open_h5, close_h5
from helpers.init import worker_init_fn
from models.baseline import get_model
from helpers.utils import mixstyle, mixup_data
from helpers import nessi
from models.baseline import initialize_weights
import optuna
torch.set_float32_matmul_precision("high")

def load_and_modify_checkpoint(pl_module,num_classes=10):
        # Modify the feed-forward layers to match the new number of classes
    pl_module.model.feed_forward[0] = nn.Conv2d(104, num_classes, kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            bias=False)
    pl_module.model.feed_forward[1] = nn.BatchNorm2d(num_classes)
    # Initialize the weights of the modified layers
    pl_module.model.feed_forward[0].apply(initialize_weights)
    pl_module.model.feed_forward[1].apply(initialize_weights)
    
    return pl_module

def load_and_modify_state_dict(ckpt_file):
    # Load the checkpoint into a dictionary
    checkpoint = torch.load(ckpt_file)
    
    # Extract the state dict from the checkpoint
    state_dict = checkpoint['state_dict']
    if state_dict is None:
        raise ValueError("Checkpoint does not contain 'state_dict'.")
    print("State dict loaded successfully.")
    # Check if 'mel.0.kernel' key is missing and add it if necessary
    if 'mel.0.kernel' not in state_dict:
        print("Adding missing 'mel.0.kernel' key with tensor size [320, 1, 459].")
        state_dict['mel.0.kernel'] = torch.randn([320, 1, 459])  # Create a random tensor with the correct size
    
    # Modify feed-forward layers to match the new number of classes
    # state_dict['model.feed_forward.0.weight'] = torch.randn([num_classes, 104, 1, 1])
    # state_dict['model.feed_forward.1.weight'] = torch.ones([num_classes])
    # state_dict['model.feed_forward.1.bias'] = torch.zeros([num_classes])
    # state_dict['model.feed_forward.1.running_mean'] = torch.zeros([num_classes])
    # state_dict['model.feed_forward.1.running_var'] = torch.ones([num_classes])
    
    return state_dict
def load_modified_checkpoint_into_pl_module(ckpt_file, config):
    # Load and modify the state dict from the checkpoint file
    modified_state_dict = load_and_modify_state_dict(ckpt_file)
    
    # Initialize the Lightning module
    pl_module = PLModule(config=config)
    if pl_module is None:
        print("pl_module is None when intializing on config")
    else:
        print("pl_module initialized successfully.")
    pl_module.load_state_dict(modified_state_dict)
    if pl_module is None:
        print("pl_module is None during load_state_dict")
    else:
        print("pl_module initialized successfully.")
    # Load the modified state dict into the Lightning module
    pl_module = load_and_modify_checkpoint(pl_module, num_classes=10)
    if pl_module is None:
        print("Final check: pl_module is None when modifying FF layers")
    else:
        print("Final check: pl_module initialized successfully.")
    return pl_module

def objective(trial):
    # Suggest temperature and KD_lambda hyperparameters for this trial
    # temperature = trial.suggest_float('temperature', 1.0,4.0)  # temperature scaling for KD
    kd_lambda = trial.suggest_float('kd_lambda', 0.02, 0.10)  # trade-off for KD loss

    # Update the config with Optuna-suggested hyperparameters
    config = argparse.Namespace(
        project_name="ICASSP_BCBL_Task1",
        experiment_name="Optuna_Trial_" + str(trial.number),
        num_workers=0,
        precision="32",
        evaluate=False,
        ckpt_id=None,
        orig_sample_rate=44100,
        subset=5,
        n_classes=10,
        in_channels=1,
        base_channels=32,
        channels_multiplier=1.8,
        expansion_rate=2.1,
        n_epochs=150,
        batch_size=256,
        mixstyle_p=0.4,
        mixstyle_alpha=0.3,
        weight_decay=0.0001,
        roll_sec=0,
        dir_prob=0.6,
        mixup_alpha=1.0,
        lr=0.01,
        warmup_steps=100,
        sample_rate=32000,
        window_length=3072,
        hop_length=500,
        n_fft=4096,
        n_mels=256,
        freqm=48,
        timem=0,
        f_min=0,
        f_max=None,
        temperature=2,
        kd_lambda=kd_lambda
    )

    # Initialize WandB logger with trial-specific settings
    wandb_logger = WandbLogger(
        project=config.project_name,
        notes="Optimization study for KD",
        tags=["DCASE24"],
        config=config,
        name=config.experiment_name
    )

    # Set up DataLoader and model as in your training function
    hf_in = open_h5('h5py_audio_wav')
    hmic_in = open_h5('h5py_mic_wav_1')
    train_dl = DataLoader(
        dataset=ntu_get_training_set_dir(config.subset, config.dir_prob, hf_in, hmic_in),
        worker_init_fn=worker_init_fn,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        pin_memory=True,
        shuffle=True
    )
    test_dl = DataLoader(
        dataset=ntu_get_test_set(hf_in),
        worker_init_fn=worker_init_fn,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        pin_memory=True
    )

    
    pl_module = PLModule(config)

    # Log model complexity (optional)
    sample = next(iter(test_dl))[0][0].unsqueeze(0)
    shape = pl_module.mel_forward(sample).size()
    macs, params = nessi.get_torch_size(pl_module.model, input_size=shape)
    wandb_logger.experiment.config['MACs'] = macs
    wandb_logger.experiment.config['Parameters'] = params

    # Trainer setup
    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        logger=wandb_logger,
        accelerator='gpu',
        devices=[0],
        num_sanity_val_steps=0,
        precision=config.precision,
        callbacks=[pl.callbacks.ModelCheckpoint(save_last=True, monitor="val/loss", save_top_k=1)]
    )

    # Train and validate
    trainer.fit(pl_module, train_dl, test_dl)
    
    # Test and retrieve metric for Optuna to optimize
    results = trainer.test(ckpt_path='best', dataloaders=test_dl)
    accuracy = results[0]['test/accuracy']  # Specify the exact metric key here
    # val_loss = trainer.callback_metrics['val/loss'].item()  # Retrieve the validation loss
    
    # Clean up
    wandb.finish()
    close_h5(hf_in)
    close_h5(hmic_in)
    
   
    
    # Return the validation loss to Optuna, which will be minimized
    # return val_loss
    # Return accuracy (or another metric) for Optuna
    return accuracy
    


## In FocusNet, we need a baseline or it's logits to adjust the weighting of the loss for student logits.
## We load logits from a .pt file by calling logits = torch.load(logits).float()


class PLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config  # results from argparse, contains all configurations for our experiment

        # module for resampling waveforms on the fly
        resample = torchaudio.transforms.Resample(
            orig_freq=self.config.orig_sample_rate,
            new_freq=self.config.sample_rate
        )

        # module to preprocess waveforms into log mel spectrograms
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.window_length,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max
        )

        freqm = torchaudio.transforms.FrequencyMasking(config.freqm, iid_masks=True)
        timem = torchaudio.transforms.TimeMasking(config.timem, iid_masks=True)

        self.mel = torch.nn.Sequential(
            resample,
            mel
        )

        self.mel_augment = torch.nn.Sequential(
            freqm,
            timem
        )
        
        # the baseline model
        self.model = get_model(n_classes=config.n_classes,
                               in_channels=config.in_channels,
                               base_channels=config.base_channels,
                               channels_multiplier=config.channels_multiplier,
                               expansion_rate=config.expansion_rate
                               )
        self.kl_div_loss = nn.KLDivLoss(log_target=True, reduction="none") # KL Divergence loss for soft, check log_target 
        self.device_ids = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']
        self.label_ids = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall',
                          'street_pedestrian', 'street_traffic', 'tram']
        # categorization of devices into 'real', 'seen' and 'unseen'
        self.device_groups = {'a': "real", 'b': "real", 'c': "real",
                              's1': "seen", 's2': "seen", 's3': "seen",
                              's4': "unseen", 's5': "unseen", 's6': "unseen"}

        # pl 2 containers:
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def mel_forward(self, x):
        """
        :param x: batch of raw audio signals (waveforms)
        :return: log mel spectrogram
        """
        x = self.mel(x)
        if self.training:
            x = self.mel_augment(x)
        x = (x + 1e-5).log()
        return x

    def forward(self, x):
        """
        :param x: batch of raw audio signals (waveforms)
        :return: final model predictions
        """
        x = self.mel_forward(x)
        x = self.model(x)
        return x

    def configure_optimizers(self):
        """
        This is the way pytorch lightening requires optimizers and learning rate schedulers to be defined.
        The specified items are used automatically in the optimization loop (no need to call optimizer.step() yourself).
        :return: optimizer and learning rate scheduler
        """
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        # schedule_lambda = \
        #     exp_warmup_linear_down(self.config.warm_up_len, self.config.ramp_down_len, self.config.ramp_down_start,
        #                            self.config.last_lr_value)
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)
        
        #For regular training
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [lr_scheduler_config]
        # return {
        #     'optimizer': optimizer,
        #     'lr_scheduler': lr_scheduler
        # }

    def training_step(self, train_batch, batch_idx):
        """
        :param train_batch: contains one batch from train dataloader
        :param batch_idx
        :return: loss to update model parameters
        """
        x, files, labels, devices, cities, teacher_logits = train_batch
        x = self.mel_forward(x)  # we convert the raw audio signals into log mel spectrograms
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device=x.device)
        if self.config.mixstyle_p > 0:
            # frequency mixstyle
            x = mixstyle(x, self.config.mixstyle_p, self.config.mixstyle_alpha)
        y_hat = self.model(x.cuda()) # This is the outputs
        # At this point we want to perform KLdiv loss      
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")
        label_loss = samples_loss.mean()
        # Temperature adjusted probabilities of teacher and student
        with torch.cuda.amp.autocast():
            y_hat_soft = F.log_softmax(y_hat / self.config.temperature, dim=-1)
            teacher_logits = F.log_softmax(teacher_logits / self.config.temperature, dim=-1)
        kd_loss = self.kl_div_loss(y_hat_soft, teacher_logits).mean()
        kd_loss = kd_loss * (self.config.temperature ** 2)
        loss = self.config.kd_lambda * label_loss + (1 - self.config.kd_lambda) * kd_loss
        # loss = kd_loss
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'])
        self.log("epoch", self.current_epoch)
        self.log("train/loss", loss)
        results = {"loss": loss, "label_loss": label_loss * self.config.kd_lambda,
                   "kd_loss": kd_loss * (1 - self.config.kd_lambda)}

        return results
        # return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, val_batch, batch_idx):
        x, files, labels, devices, cities = val_batch
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device=x.device)
        y_hat = self.forward(x.cuda())
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")

        # for computing accuracy
        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()

        dev_names = [d.rsplit("-", 1)[1][:-4] for d in files]
        results = {'loss': samples_loss.mean(), "n_correct": n_correct,
                   "n_pred": torch.as_tensor(len(labels), device=self.device)}

        '''# log metric per device and scene
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
        results = {k: v.detach() for k, v in results.items()}'''
        self.validation_step_outputs.append(results)

    def on_validation_epoch_end(self):
        # convert a list of dicts to a flattened dict
        outputs = {k: [] for k in self.validation_step_outputs[0]}
        for step_output in self.validation_step_outputs:
            for k in step_output:
                outputs[k].append(step_output[k])
        for k in outputs:
            outputs[k] = torch.stack(outputs[k])

        avg_loss = outputs['loss'].mean()
        acc = sum(outputs['n_correct']) * 1.0 / sum(outputs['n_pred'])

        logs = {'acc': acc, 'loss': avg_loss}

        '''# log metric per device and scene
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
        # prefix with 'val' for logging'''
        self.log_dict({"val/" + k: logs[k] for k in logs})
        self.validation_step_outputs.clear()

    def test_step(self, test_batch, batch_idx):
        x, files, labels, devices, cities = test_batch
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device=x.device)
        # maximum memory allowance for parameters: 128 KB
        # baseline has 61148 parameters -> we can afford 16-bit precision
        # since 61148 * 16 bit ~ 122 kB

        # assure fp16
        self.model.half()
        x = self.mel_forward(x)
        x = x.half()
        y_hat = self.model(x.cuda())
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")

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

    def predict_step(self, eval_batch, batch_idx, dataloader_idx=0):
        x, files = eval_batch

        # assure fp16
        self.model.half()

        x = self.mel_forward(x)
        x = x.half()
        y_hat = self.model(x)

        return files, y_hat
# Running the Optuna study
if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    print("  Value:", study.best_trial.value)
    print("  Params:", study.best_trial.params)
    