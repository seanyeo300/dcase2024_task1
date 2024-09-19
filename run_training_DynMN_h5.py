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

from dataset.dcase24_ntu_DynMN import ntu_get_training_set_dir, ntu_get_test_set, ntu_get_eval_set, open_h5, close_h5
# from models.mn.model import get_model as get_mobilenet
from models.dymn.model import get_model as get_dymn
from models.preprocess import AugmentMelSTFT
from helpers.init import worker_init_fn
from helpers.utils import NAME_TO_WIDTH, exp_warmup_linear_down, mixup, mixstyle


def train(config):
    # Train Models for Acoustic Scene Classification

   # logging is done using wandb
    wandb_logger = WandbLogger(
        project=config.project_name,
        notes="Baseline System for DCASE'24 Task 1.",
        tags=["DCASE24"],
        config=config,  # this logs all hyperparameters for us
        name=config.experiment_name
    )
    # train dataloader
    assert config.subset in {100, 50, 25, 10, 5}, "Specify an integer value in: {100, 50, 25, 10, 5} to use one of " \
                                                  "the given subsets."
    # get pointer to h5 file containing audio samples
    hf_in = open_h5('h5py_audio_wav')
    hmic_in = open_h5('h5py_mic_wav_1')
    device = torch.device('cuda') if config.cuda and torch.cuda.is_available() else torch.device('cpu')

    # model to preprocess waveform into mel spectrograms
    mel = AugmentMelSTFT(n_mels=config.n_mels,
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
    mel.to(device)

    # load prediction model
    model_name = config.model_name
    pretrained_name = model_name if config.pretrained else None
    width = NAME_TO_WIDTH(model_name) if model_name and config.pretrained else config.model_width
    if model_name.startswith("dymn"):
        model = get_dymn(width_mult=width, pretrained_name=pretrained_name,
                         pretrain_final_temp=config.pretrain_final_temp,
                         num_classes=10)
    # else:
    #     model = get_mobilenet(width_mult=width, pretrained_name=pretrained_name,
    #                           head_type=config.head_type, se_dims=config.se_dims,
    #                           num_classes=10)
    model.to(device)

    # dataloader
    dl = DataLoader(dataset=ntu_get_training_set_dir(config.subset, config.dir_prob, config.gain_augment, hf_in, hmic_in),               
                          worker_init_fn=worker_init_fn,
                          num_workers=config.num_workers,
                          batch_size=config.batch_size,
                          pin_memory= True,
                          shuffle=True)
    # evaluation loader
    eval_dl = DataLoader(dataset=ntu_get_test_set(config.cache_path, config.resample_rate),
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)

    # optimizer & scheduler
    lr = config.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # phases of lr schedule: exponential increase, constant lr, linear decrease, fine-tune
    schedule_lambda = \
        exp_warmup_linear_down(config.warm_up_len, config.ramp_down_len, config.ramp_down_start, config.last_lr_value)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)

    name = None
    accuracy, val_loss = float('NaN'), float('NaN')

    for epoch in range(config.n_epochs):
        mel.train()
        model.train()
        train_stats = dict(train_loss=list())
        pbar = tqdm(dl)
        pbar.set_description("Epoch {}/{}: accuracy: {:.4f}, val_loss: {:.4f}"
                             .format(epoch + 1, config.n_epochs, accuracy, val_loss))
        for batch in pbar:
            x, f, y, dev, city, index = batch
            bs = x.size(0)
            x, y = x.to(device), y.to(device)
            x = _mel_forward(x, mel)

            if config.mixstyle_p > 0:
                x = mixstyle(x, config.mixstyle_p, config.mixstyle_alpha)
                y_hat, _ = model(x)
                samples_loss = F.cross_entropy(y_hat, y, reduction="none")
            elif config.mixup_alpha:
                rn_indices, lam = mixup(bs, config.mixup_alpha)
                lam = lam.to(x.device)
                x = x * lam.reshape(bs, 1, 1, 1) + \
                    x[rn_indices] * (1. - lam.reshape(bs, 1, 1, 1))
                y_hat, _ = model(x)
                samples_loss = (F.cross_entropy(y_hat, y, reduction="none") * lam.reshape(bs) +
                                F.cross_entropy(y_hat, y[rn_indices], reduction="none") * (
                                            1. - lam.reshape(bs)))

            else:
                y_hat, _ = model(x)
                samples_loss = F.cross_entropy(y_hat, y, reduction="none")

            # loss
            loss = samples_loss.mean()

            # append training statistics
            train_stats['train_loss'].append(loss.detach().cpu().numpy())

            # Update Model
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Update learning rate
        scheduler.step()

        # evaluate
        accuracy, val_loss = _test(model, mel, eval_dl, device)

        # log train and validation statistics
        wandb.log({"train_loss": np.mean(train_stats['train_loss']),
                   "accuracy": accuracy,
                   "val_loss": val_loss
                   })

        # remove previous model (we try to not flood your hard disk) and save latest model
        if name is not None:
            os.remove(os.path.join(wandb.run.dir, name))
        name = f"mn{str(width).replace('.', '')}_dcase_epoch_{epoch}_acc_{int(round(accuracy*1000))}.pt"
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, name))


def _mel_forward(x, mel):
    old_shape = x.size()
    x = x.reshape(-1, old_shape[2])
    x = mel(x)
    x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
    return x


def _test(model, mel, eval_loader, device):
    model.eval()
    mel.eval()

    targets = []
    outputs = []
    losses = []
    pbar = tqdm(eval_loader)
    pbar.set_description("Validating")
    for batch in pbar:
        x, f, y, dev, city, index = batch
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            x = _mel_forward(x, mel)
            y_hat, _ = model(x)
        targets.append(y.cpu().numpy())
        outputs.append(y_hat.float().cpu().numpy())
        losses.append(F.cross_entropy(y_hat, y).cpu().numpy())

    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)
    losses = np.stack(losses)
    accuracy = metrics.accuracy_score(targets, outputs.argmax(axis=1))
    return accuracy, losses.mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')

    # general
    parser.add_argument('--experiment_name', type=str, default="tDynMN_32K_FMS_DIR")
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--cache_path', type=str, default=None)

    # training
    parser.add_argument('--pretrained', action='store_true', default=True) # Pre-trained on AS
    parser.add_argument('--model_name', type=str, default="dymn20_as") # Best MAP model
    parser.add_argument('--pretrain_final_temp', type=float, default=1.0)  # for DyMN
    parser.add_argument('--model_width', type=float, default=1.0)
    parser.add_argument('--head_type', type=str, default="mlp")
    parser.add_argument('--se_dims', type=str, default="c")
    parser.add_argument('--n_epochs', type=int, default=80)
    parser.add_argument('--mixup_alpha', type=float, default=0.3)
    parser.add_argument('--mixstyle_p', type=float, default=0.0)
    parser.add_argument('--mixstyle_alpha', type=float, default=0.4)
    parser.add_argument('--no_roll', action='store_true', default=False)
    parser.add_argument('--no_wavmix', action='store_true', default=False)
    parser.add_argument('--gain_augment', type=int, default=12)
    parser.add_argument('--weight_decay', type=int, default=0.0)

    # lr schedule
    parser.add_argument('--lr', type=float, default=8e-4)
    parser.add_argument('--warm_up_len', type=int, default=10)
    parser.add_argument('--ramp_down_start', type=int, default=10)
    parser.add_argument('--ramp_down_len', type=int, default=65)
    parser.add_argument('--last_lr_value', type=float, default=0.01)

    # preprocessing
    parser.add_argument('--resample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=800)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--freqm', type=int, default=0)
    parser.add_argument('--timem', type=int, default=0)
    parser.add_argument('--fmin', type=int, default=0)
    parser.add_argument('--fmax', type=int, default=None)
    parser.add_argument('--fmin_aug_range', type=int, default=10)
    parser.add_argument('--fmax_aug_range', type=int, default=2000)

    args = parser.parse_args()
    train(args)
