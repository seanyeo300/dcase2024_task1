#### This is the script to generate h5 files provided by Joseph####

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
# import torch
import torchaudio
from torch.utils.data import DataLoader, BatchSampler
# import argparse
import torch.nn.functional as F
# import transformers
# import wandb
# import json

# from dataset.dcase24 import get_training_set, get_test_set, get_eval_set
# from dataset.ntu_dcase24_v1 import ntu_get_training_set, ntu_get_training_set_dir, ntu_get_test_set, ntu_gen_base_training_h5, open_h5, close_h5
# from helpers.init import worker_init_fn
# from models.baseline import get_model
# # from models.ntu_baseline import get_ntu_model, get_ntu_protonet
# from helpers.utils import mixstyle, ntu_mixstyle
# from helpers import nessi

# import shutil
# import pathlib
import os

# from helpers.ntu_data_sampler import ProtoNetDataSampler
# from dataset.ntu_dcase24 import SimpleSelectionDataset, BasicDCASE24DatasetProtoNet
import pandas as pd

from argparse import Namespace
import h5py
from tqdm import tqdm
from models.mel import AugmentMelSTFT 
# import glob

args = Namespace(project_name='DCASE24_BCBL', 
                    experiment_name='Protonet-NTU', 
                    num_workers=0, precision='32', 
                    evaluate=False, ckpt_id='057o1jd1', 
                    orig_sample_rate=44100, 
                    subset=5, n_classes=10, 
                    in_channels=1, 
                    base_channels=20, 
                    channels_multiplier=1.5, 
                    expansion_rate=2, 
                    n_epochs=1, 
                    batch_size=256, 
                    mixstyle_p=0.4, 
                    mixstyle_alpha=0.3, 
                    weight_decay=0.00001, 
                    roll_sec=0, lr=0.01, 
                    dir_p = 0.6,
                    warmup_steps=20, 
                    sample_rate=44100,
                    resample_rate = 44100, 
                    window_length=3072, 
                    hop_length=500, 
                    n_fft=4096, 
                    n_mels=256, 
                    freqm=0, 
                    timem=0, 
                    f_min=0, 
                    f_max=None,
                    fmin_aug_range=1,
                    fmax_aug_range=1000,
                    way=10, 
                    shot=5, 
                    query=10, 
                    episode=1000)

# args = Namespace(project_name='DCASE24_Task1_PaSST', 
#                     experiment_name='PaSST-NTU', 
#                     num_workers=0, precision='32', 
#                     evaluate=False, ckpt_id='057o1jd1', 
#                     orig_sample_rate=44100,
#                     subset=5, n_classes=13, 
#                     in_channels=1, 
#                     base_channels=20, 
#                     channels_multiplier=1.5, 
#                     expansion_rate=2, 
#                     n_epochs=1, 
#                     batch_size=256, 
#                     mixstyle_p=0.4, 
#                     mixstyle_alpha=0.3, 
#                     weight_decay=0.00001, 
#                     roll_sec=0, lr=0.01, 
#                     dir_p = 0.6,
#                     warmup_steps=20,##################### Start Here #####################
#                     resample_rate = 44100,
#                     sample_rate=44100, 
#                     window_length=800, 
#                     hop_length=320, 
#                     n_fft=1024, 
#                     n_mels=128,
#                     freqm=0, 
#                     timem=0, ##################### Done ######################## 
#                     f_min=0, 
#                     f_max=None,
#                     fmin_aug_range=1,
#                     fmax_aug_range=1000,
#                     way=10, 
#                     shot=5, 
#                     query=10, 
#                     episode=1000)
config = args
# meta_csv = r"F:\DCASE\2024\Datasets\TAU-urban-acoustic-scenes-2022-mobile-development\meta.csv" # DSP
meta_csv = r"D:\Sean\DCASE\datasets\Extract_to_Folder\TAU-urban-acoustic-scenes-2022-mobile-development\meta.csv" # ALI
# meta_csv = r"F:\CochlScene\meta.csv" # DSP
# meta_csv = r"D:\Sean\CochlScene\meta.csv" # ALI
# train_files_csv = r"F:\DCASE\2024\Datasets\TAU-urban-acoustic-scenes-2022-mobile-development\split100.csv" # DSP
train_files_csv = r"D:\Sean\DCASE\datasets\Extract_to_Folder\TAU-urban-acoustic-scenes-2022-mobile-development\split100.csv" # ALI
# train_files_csv = r"F:\Github\dcase2024_task1\split_setup\splitcochl.csv"
# train_files_csv = r"D:\Sean\github\cpjku_dcase23_NTU\split_setup\splitcochl10s.csv"
# eval_meta_csv = 'c:/Dataset/eval_dataset_2024/meta.csv'
# dataset_dir = r"F:\DCASE\2024\Datasets\TAU-urban-acoustic-scenes-2022-mobile-development" # DSP
dataset_dir = r"D:\Sean\DCASE\datasets\Extract_to_Folder\TAU-urban-acoustic-scenes-2022-mobile-development" # ALI
# dataset_dir = r"F:\CochlScene"
# dataset_dir = r"D:\Sean\CochlScene" # ALI
# eval_dataset_dir = 'c:/Dataset/eva_dataset_2024/'
# eval_csv = r"F:\DCASE\2024\Datasets\TAU-urban-acoustic-scenes-2024-mobile-evaluation\evaluation_setup\fold1_test.csv" # DSP
eval_csv = r"D:\Sean\DCASE\datasets\Extract_to_Folder\TAU-urban-acoustic-scenes-2022-mobile-development\evaluation_setup\fold1_test.csv" #ALI
# eval_csv = r"F:\Github\dcase2024_task1\split_setup\val_cochl10s.csv" #DSP
# eval_csv = r"D:\Sean\github\cpjku_dcase23_NTU\split_setup\val_cochl10s.csv" # ALI
# eval_dir = r"F:\DCASE\2024\Datasets\TAU-urban-acoustic-scenes-2024-mobile-evaluation" # DSP
# eval_dir = r"F:\DCASE\2024\Datasets\TAU-urban-acoustic-scenes-2024-mobile-evaluation" # AlI
# test_csv = r"F:\Github\dcase2024_task1\split_setup\test_cochl10s.csv" # DSP
test_csv = r"D:\Sean\github\dcase2024_task1\split_setup\test.csv" # ALI
# test_csv = r"D:\Sean\github\cpjku_dcase23_NTU\split_setup\test_cochl10s.csv" # ALI
# eval_dir = r"F:\DCASE\2024\Datasets\TAU-urban-acoustic-scenes-2024-mobile-evaluation"


dataset_config = {
    "dataset_name": "tau24",
    "meta_csv": os.path.join(dataset_dir, "meta.csv"),
    "split_path": "split_setup",
    "split_url": "https://github.com/CPJKU/dcase2024_task1_baseline/releases/download/files/",
    "test_split_csv": "test.csv",
    "eval_dir": os.path.join(dataset_dir, "..", "eval_dataset_2024"),
    "eval_meta_csv": os.path.join(dataset_dir, "..", "eval_dataset_2024", "meta.csv"),
    "dirs_path": 'C:/Dataset/mic_Impulse/',    
}

mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=config.sample_rate,
    n_fft=config.n_fft,
    win_length=config.window_length,
    hop_length=config.hop_length,
    n_mels=config.n_mels,
    f_min=config.f_min,
    f_max=config.f_max
    )

mel_passt = AugmentMelSTFT(n_mels=config.n_mels,
                                  sr=config.resample_rate,
                                  win_length=config.window_length,
                                  hopsize=config.hop_length,
                                  n_fft=config.n_fft,
                                  freqm=config.freqm,
                                  timem=config.timem,
                                  fmin=config.f_min,
                                  fmax=config.f_max,
                                  fmin_aug_range=config.fmin_aug_range,
                                  fmax_aug_range=config.fmax_aug_range
                                  )


# # to create audio samples h5 file
<<<<<<< Updated upstream
df = pd.read_csv(meta_csv, sep="\t")
train_files = pd.read_csv(train_files_csv, sep='\t')['filename'].values.reshape(-1)
files = df['filename'].values.reshape(-1)

hf = h5py.File('h5py_cochl_wav', 'w')
for file_idx in tqdm(range(len(files))):
    mel_sig, _ = torchaudio.load(os.path.join(dataset_dir, files[file_idx]))
    #output_str = dataset_dir + 'h5' + train_files[file_idx][5:-4] + '.h5'
    output_str = files[file_idx][5:-4]
    print(f"output = {output_str}")
    #with h5py.File(output_str, 'w') as hf:
    hf.create_dataset(output_str, data = mel_sig)    
hf.close()

# Create mel HDF5 file
# df = pd.read_csv(meta_csv, sep="\t")
# train_files = pd.read_csv(train_files_csv, sep='\t')['filename'].values.reshape(-1)
# files = df['filename'].values.reshape(-1)

# hf = h5py.File('h5py_cochl10_256bins', 'w')
# for file_idx in tqdm(range(len(files))):
#     sig, _ = torchaudio.load(os.path.join(dataset_dir, files[file_idx]))
#     mel_sig = mel_passt(sig)
#     output_str = files[file_idx][5:-4]
#     hf.create_dataset(output_str, data=mel_sig)
# hf.close()



=======
# df = pd.read_csv(meta_csv, sep="\t")
# train_files = pd.read_csv(train_files_csv, sep='\t')['filename'].values.reshape(-1)
# files = df['filename'].values.reshape(-1)

# hf = h5py.File('h5py_audio_wav_test', 'w')
# for file_idx in tqdm(range(len(files))):
#     mel_sig, _ = torchaudio.load(os.path.join(dataset_dir, files[file_idx]),format='wav')
#     #output_str = dataset_dir + 'h5' + train_files[file_idx][5:-4] + '.h5'
#     output_str = files[file_idx][5:-4]
#     # print(f"output = {output_str}")
#     #with h5py.File(output_str, 'w') as hf:
#     hf.create_dataset(output_str, data = mel_sig)    
# hf.close()

# Create mel HDF5 file
# df = pd.read_csv(meta_csv, sep="\t")
# train_files = pd.read_csv(train_files_csv, sep='\t')['filename'].values.reshape(-1)
# files = df['filename'].values.reshape(-1)

# hf = h5py.File('h5py_cochl10_256bins', 'w')
# for file_idx in tqdm(range(len(files))):
#     sig, _ = torchaudio.load(os.path.join(dataset_dir, files[file_idx]))
#     mel_sig = mel_passt(sig)
#     output_str = files[file_idx][5:-4]
#     hf.create_dataset(output_str, data=mel_sig)
# hf.close()

dir_folder = r'F:\Github\dcase2024_task1\dataset\dirs'
# Create a list of all wav files in the directory
files = [f for f in os.listdir(dir_folder) if f.endswith('.wav')]
hf = h5py.File('h5py_mic_wav_1.h5', 'w')

# Iterate over each file in the directory
for file_idx in tqdm(range(len(files))):
    # Load the audio signal from the file
    mel_sig, _ = torchaudio.load(os.path.join(dir_folder, files[file_idx]))
    # Generate the output string for the dataset name
    output_str = files[file_idx][:-4]  # Removing '.wav' extension
    # Create a dataset in the HDF5 file with the audio signal data
    hf.create_dataset(output_str, data=mel_sig)

# Close the HDF5 file
hf.close()

>>>>>>> Stashed changes
# # to create mel data h5 file
# df = pd.read_csv(meta_csv, sep="\t")
# #train_files = pd.read_csv(train_files_csv, sep='\t')['filename'].values.reshape(-1)
# train_files = pd.read_csv(train_files_csv, sep='\t')['filename'].values.reshape(-1)
# files = df[['filename']].values.reshape(-1)

# hf = h5py.File('h5py_mel_256bins', 'w')
# for file_idx in tqdm(range(len(files))):
#     sig, _ = torchaudio.load(os.path.join(dataset_dir, files[file_idx]))
#     mel_sig = mel(sig)
#     #output_str = dataset_dir + 'h5' + train_files[file_idx][5:-4] + '.h5'
#     output_str = files[file_idx][5:-4]
#     #with h5py.File(output_str, 'w') as hf:
#     hf.create_dataset(output_str, data = mel_sig)    
# hf.close() 



# # Save eval set files to h5
# df = pd.read_csv(eval_csv, sep="\t")
# # train_files = pd.read_csv(train_files_csv, sep='\t')['filename'].values.reshape(-1)
# files = df['filename'].values.reshape(-1)

# hf = h5py.File('h5py_audio_eval_wav', 'w')
# for file_idx in tqdm(range(len(files))):
#     mel_sig, _ = torchaudio.load(os.path.join(eval_dir, files[file_idx]))
#     #output_str = dataset_dir + 'h5' + train_files[file_idx][5:-4] + '.h5'
#     output_str = files[file_idx][5:-4]
#     #with h5py.File(output_str, 'w') as hf:
#     hf.create_dataset(output_str, data = mel_sig)    
# hf.close()