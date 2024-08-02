import pandas as pd
import os
from sklearn import preprocessing
from torch.utils.data import Dataset as TorchDataset
import torch
import torchaudio
from torch.hub import download_url_to_file
import numpy as np
#from io import BytesIO
import h5py
from scipy.signal import convolve

dataset_dir = 'c:/Dataset/dataset/'
dataset_eval_dir = 'c:/Dataset/eva_dataset_2024/'
assert dataset_dir is not None, "Specify 'TAU Urban Acoustic Scenes 2022 Mobile dataset' location in variable " \
                                "'dataset_dir'. The dataset can be downloaded from this URL:" \
                                " https://zenodo.org/record/6337421"


dataset_config = {
    "dataset_name": "tau24",
    "meta_csv": os.path.join(dataset_dir, "meta.csv"),
    "split_path": "split_setup",
    "split_url": "https://github.com/CPJKU/dcase2024_task1_baseline/releases/download/files/",
    "test_split_csv": "test.csv",
    "eval_dir": dataset_eval_dir,
    "eval_meta_csv": os.path.join(dataset_eval_dir, "meta.csv"),
    "dirs_path": 'C:/Dataset/mic_Impulse/',    
}

class BasicDCASE24Dataseth5(TorchDataset):
    """
    Basic DCASE'24 Dataset: loads mel data from files
    """

    def __init__(self, meta_csv, hf_in):
        """
        @param meta_csv: meta csv file for the dataset
        return: waveform, file, label, device and city
        """
        df = pd.read_csv(meta_csv, sep="\t")
        le = preprocessing.LabelEncoder()
        self.labels = torch.from_numpy(le.fit_transform(df[['scene_label']].values.reshape(-1)))
        self.devices = le.fit_transform(df[['source_label']].values.reshape(-1))
        self.cities = le.fit_transform(df['identifier'].apply(lambda loc: loc.split("-")[0]).values.reshape(-1))
        self.files = df[['filename']].values.reshape(-1)
        self.hf_in = hf_in

    def __getitem__(self, index):
        mel_sig_ds = self.files[index][5:-4]
        sig = torch.from_numpy(self.hf_in.get(mel_sig_ds)[()])  
        return sig, self.files[index], self.labels[index], self.devices[index], self.cities[index]

    def __len__(self):
        return len(self.files)   

class SimpleSelectionDataset(TorchDataset):
    """A dataset that selects a subsample from a dataset based on a set of sample ids.
        Supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, dataset, available_indices):
        """
        @param dataset: dataset to load data from
        @param available_indices: available indices of samples for different splits
        return: waveform, file, label, device, city
        """
        self.available_indices = available_indices
        self.dataset = dataset

    def __getitem__(self, index):
        x, file, label, device, city = self.dataset[self.available_indices[index]]
        return x, file, label, device, city

    def __len__(self):
        return len(self.available_indices)
    
 

class RollDataset(TorchDataset):
    """A dataset implementing time rolling of waveforms.
    """

    def __init__(self, dataset: TorchDataset, shift_range: int, axis=1):
        """
        @param dataset: dataset to load data from
        @param shift_range: maximum shift range
        return: waveform, file, label, device, city
        """
        self.dataset = dataset
        self.shift_range = shift_range
        self.axis = axis

    def __getitem__(self, index):
        x, file, label, device, city = self.dataset[index]
        sf = int(np.random.random_integers(-self.shift_range, self.shift_range))
        return x.roll(sf, self.axis), file, label, device, city

    def __len__(self):
        return len(self.dataset)
    
class DirDataset(TorchDataset):
    """
   Augments Waveforms with a Device Impulse Response (DIR)
    """

    def __init__(self, ds, hmic, dir_p):
        self.ds = ds
        self.hmic = hmic
        self.dir_p = dir_p

    def __getitem__(self, index):
        x, file, label, device, city = self.ds[index]

        self.device = device

        # New devices are created using device A + impulse function + DRC
        if self.device == 'a' and self.dir_p > np.random.rand():
            # choose a DIR at random
            dir_idx = str(int(np.random.randint(0, len(self.hmic))))
            dir = torch.from_numpy(self.hmic.get(dir_idx)[()])  
            # get audio file with 'new' mic response
            x = convolve(x, dir, 'full')[:, :x.shape[1]]
            x = torch.from_numpy(x)
        return x, file, label, device, city

    def __len__(self):
        return len(self.ds)  


def ntu_get_training_set_dir(split=100, dir_prob = False, hf_in=None, hmic_in=None):
    assert str(split) in ("5", "10", "25", "50", "100"), "Parameters 'split' must be in [5, 10, 25, 50, 100]"
    os.makedirs(dataset_config['split_path'], exist_ok=True)
    subset_fname = f"split{split}.csv"
    subset_split_file = os.path.join(dataset_config['split_path'], subset_fname)
    if not os.path.isfile(subset_split_file):
        # download split{x}.csv (file containing all audio snippets for respective development-train split)
        subset_csv_url = dataset_config['split_url'] + subset_fname
        print(f"Downloading file: {subset_fname}")
        download_url_to_file(subset_csv_url, subset_split_file)
    ds = ntu_get_base_training_set(dataset_config['meta_csv'], subset_split_file, hf_in)
    if dir_prob:
        ds = DirDataset(ds, hmic_in, dir_prob)
    return ds


def ntu_get_base_training_set(meta_csv, train_files_csv, hf_in):
    meta = pd.read_csv(meta_csv, sep="\t")
    train_files = pd.read_csv(train_files_csv, sep='\t')['filename'].values.reshape(-1)
    train_subset_indices = list(meta[meta['filename'].isin(train_files)].index)
    ds = SimpleSelectionDataset(BasicDCASE24Dataseth5(meta_csv, hf_in),
                                train_subset_indices)
    return ds

def ntu_get_test_set(hf_in = None):
    os.makedirs(dataset_config['split_path'], exist_ok=True)
    test_split_csv = os.path.join(dataset_config['split_path'], dataset_config['test_split_csv'])
    if not os.path.isfile(test_split_csv):
        # download test.csv (file containing all audio snippets for development-test split)
        test_csv_url = dataset_config['split_url'] + dataset_config['test_split_csv']
        print(f"Downloading file: {dataset_config['test_split_csv']}")
        download_url_to_file(test_csv_url, test_split_csv)
    ds = ntu_get_base_test_set(dataset_config['meta_csv'], test_split_csv, hf_in)
    return ds

def ntu_get_base_test_set(meta_csv, test_files_csv, hf_in):
    meta = pd.read_csv(meta_csv, sep="\t")
    test_files = pd.read_csv(test_files_csv, sep='\t')['filename'].values.reshape(-1)
    test_indices = list(meta[meta['filename'].isin(test_files)].index)
    ds = SimpleSelectionDataset(BasicDCASE24Dataseth5(meta_csv, hf_in), test_indices)
    return ds


class BasicDCASE24EvalDataset(TorchDataset):
    """
    Basic DCASE'24 Dataset: loads eval data from files
    """

    def __init__(self, meta_csv, eval_dir):
        """
        @param meta_csv: meta csv file for the dataset
        @param eval_dir: directory containing evaluation set
        return: waveform, file
        """
        df = pd.read_csv(meta_csv, sep="\t")
        self.files = df[['filename']].values.reshape(-1)
        self.eval_dir = eval_dir

    def __getitem__(self, index):
        sig, _ = torchaudio.load(os.path.join(self.eval_dir, self.files[index]))
        return sig, self.files[index]

    def __len__(self):
        return len(self.files)
    
class BasicDCASE24EvalDataseth5(TorchDataset):
    """
    Basic DCASE'24 Dataset: loads eval data from files
    """

    def __init__(self, meta_csv, hf_in):
        """
        @param meta_csv: meta csv file for the dataset
        @param eval_dir: directory containing evaluation set
        return: waveform, file
        """
        df = pd.read_csv(meta_csv, sep="\t")
        self.files = df[['filename']].values.reshape(-1)
        self.hf_in = hf_in
        #self.eval_dir = eval_dir

    def __getitem__(self, index):
        mel_sig_ds = self.files[index]
        sig = torch.from_numpy(self.hf_in.get(mel_sig_ds)[()])          
        return sig, self.files[index]

    def __len__(self):
        return len(self.files)    
    

def get_eval_set():
    assert os.path.exists(dataset_config['eval_dir']), f"No such folder: {dataset_config['eval_dir']}"
    ds = get_base_eval_set(dataset_config['eval_meta_csv'], dataset_config['eval_dir'])
    return ds

def get_base_eval_set(meta_csv, eval_dir):
    ds = BasicDCASE24EvalDataset(meta_csv, eval_dir)
    return ds

def ntu_get_eval_set(hf_in):
    assert os.path.exists(dataset_config['eval_dir']), f"No such folder: {dataset_config['eval_dir']}"
    ds = ntu_get_base_eval_set(dataset_config['eval_meta_csv'], hf_in)
    return ds

def ntu_get_base_eval_set(meta_csv, hf_in):
    ds = BasicDCASE24EvalDataseth5(meta_csv, hf_in)
    return ds

def ntu_gen_base_training_h5(config):
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        win_length=config.window_length,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        f_min=config.f_min,
        f_max=config.f_max
    )    
    train_files = pd.read_csv(dataset_config['meta_csv'], sep='\t')['filename'].values.reshape(-1)

    hf = h5py.File('h5py_mel', 'w')
    for file_idx in range(len(train_files)):
        sig, _ = torchaudio.load(os.path.join(dataset_dir, train_files[file_idx]))
        mel_sig = mel(sig)
        output_str = train_files[file_idx][6:-4]
        hf.create_dataset(output_str, data = mel_sig)    
    hf.close()    

def open_h5(h5_file):
    hf_in =h5py.File(h5_file, 'r')
    return hf_in

def close_h5(hf_in):
    hf_in.close()    

def open_mic_h5(h5_file):
    hf_in =h5py.File(h5_file, 'r')
    return hf_in

def close_mic_h5(hf_in):
    hf_in.close() 

def open_audio_h5(h5_file):
    hf_in =h5py.File(h5_file, 'r')
    return hf_in

def close_audio_h5(hf_in):
    hf_in.close()                