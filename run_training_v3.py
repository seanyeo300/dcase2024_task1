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

from dataset.dcase24 import get_training_set, get_test_set, get_eval_set
from dataset.ntu_dcase24_v1 import ntu_get_training_set, ntu_get_training_set_dir, ntu_get_test_set, open_h5, close_h5
from helpers.init import worker_init_fn
from models.baseline import get_model
from models.ntu_baseline import get_ntu_model, get_ntu_model_no_gru
from helpers.utils import mixstyle, ntu_mixstyle, mixstyle_1
from helpers import nessi

import shutil
import pathlib
import os
from torch.autograd import Variable

os.environ["WANDB_SILENT"] = "true"

#from pytorch_lightning.strategies import DeepSpeedStrategy

def train(config):
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

    # convert audio files to melspectrogram before training
    #print('Starting audio to mel conversion for all files')
    #ntu_gen_base_training_h5(config)
    #print('Finish audio to mel conversion')

    # get pointer to h5 file containing audio samples
    hf_in = open_h5('h5py_audio_wav')
    #hf_in = open_h5('h5py_mel_192bins')
    # get pointer to h5 file containing impulse responses
    hmic_in = open_h5('h5py_mic_wav_1')

    #roll_samples = config.orig_sample_rate * config.roll_sec
    #train_dl = DataLoader(dataset=ntu_get_training_set(config.subset, hf_in),
    train_dl = DataLoader(dataset=ntu_get_training_set_dir(config.subset, config.dir_p, hf_in, hmic_in),               
                          worker_init_fn=worker_init_fn,
                          num_workers=config.num_workers,
                          batch_size=config.batch_size,
                          pin_memory= True,
                          shuffle=True)

    test_dl = DataLoader(dataset=ntu_get_test_set(hf_in),
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size,
                         pin_memory=True)
    
    # create pytorch lightening module
    pl_module = PLModule(config)

    # get model complexity from nessi and log results to wandb
    sample = next(iter(test_dl))[0][0].unsqueeze(0)
    shape = pl_module.mel_forward(sample).size()
    macs, params = nessi.get_torch_size(pl_module.model, input_size=shape)
    # log MACs and number of parameters for our model
    wandb_logger.experiment.config['MACs'] = macs
    wandb_logger.experiment.config['Parameters'] = params
    save_dir = os.path.abspath(os.getcwd()) + '/' + config.project_name + '/' + wandb.run.id +'/'
    os.makedirs(save_dir)
    shutil.copy2(os.path.basename(__file__), save_dir)
    shutil.copy2('C:/BL1/models/ntu_baseline.py', save_dir)     

    #checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=1, monitor="val/macro_avg_acc", mode="max")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=1, monitor="val/acc", mode="max")
    #progress_bar_cb = pl.callbacks.RichProgressBar(leave = True)
    # create the pytorch lightening trainer by specifying the number of epochs to train, the logger,
    # on which kind of device(s) to train and possible callbacks
    #strategy = DeepSpeedStrategy()
    trainer = pl.Trainer(max_epochs=config.n_epochs,
                         num_sanity_val_steps=0,
                         logger=wandb_logger,
                         accelerator='gpu',
                         devices=1,
                         precision=config.precision,
                         log_every_n_steps = 28,                         
                         callbacks=[checkpoint_callback])
    # start training and validation for the specified number of epochs
    trainer.fit(pl_module, train_dataloaders = train_dl, val_dataloaders = test_dl)

    # get checkpoint path 
    ckp_path = pathlib.Path(checkpoint_callback.best_model_path)

    # final test step
    # here: use the validation split
    trainer.test(ckpt_path=ckp_path, dataloaders=test_dl)
    # close file pointer to h5 file 
    close_h5(hf_in)
    close_h5(hmic_in)
 
    # backup main and model files
    #shutil.copy2(os.path.basename(__file__), ckp_path.parent)
    #shutil.copy2('C:/BL1/models/ntu_baseline.py', ckp_path.parent)
    wandb.finish()

