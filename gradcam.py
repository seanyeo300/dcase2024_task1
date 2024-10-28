import torch
import torchaudio
import argparse
import os
import pytorch_lightning as pl
from pytorch_grad_cam import GradCAM
import matplotlib.pyplot as plt
from models.passt import get_model
from models.mel import AugmentMelSTFT

# Define the PLModule class
class PLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mel = AugmentMelSTFT(
            n_mels=config.n_mels,
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
        self.model = get_model(
            arch="passt_s_swa_p16_128_ap476",
            n_classes=config.n_classes,
            input_fdim=config.input_fdim,
            s_patchout_t=config.s_patchout_t,
            s_patchout_f=config.s_patchout_f
        )
        # Additional attributes can be defined here

# Function to perform inference and visualize Grad-CAM
def perform_inference_and_visualize(pl_module, audio_path):
    waveform, sr = torchaudio.load(audio_path)
    print(f"Loaded waveform shape: {waveform.shape}, Sample rate: {sr}")
    if sr != pl_module.config.resample_rate:
        resampler = torchaudio.transforms.Resample(sr, pl_module.config.resample_rate)
        waveform = resampler(waveform)

    # Compute mel spectrogram using the model's mel transformation
    mel_spectrogram = pl_module.mel(waveform)  # Directly use the waveform as input
    print(f"Mel spectrogram shape: {mel_spectrogram.shape}")
    with torch.no_grad():
        output = pl_module.model(mel_spectrogram.unsqueeze(0))  # Pass through the model
        print(f"Model output shape: {output.shape}")
    target_layer = pl_module.model.head[-1]  # Adjust based on the model's architecture
    print(f"Using target layer: {target_layer}")
    cam = GradCAM(model=pl_module.model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())  # Correctly define cam
    grayscale_cam = cam(input_tensor=mel_spectrogram.unsqueeze(0))[0]

    plt.imshow(grayscale_cam, cmap="jet")
    plt.colorbar()
    plt.title("Class Activation Map (CAM)")
    plt.show()


   # Main script with argument parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference with CAM visualization')
    
    # General arguments
    parser.add_argument('--project_name', type=str, default="NTU24_ASC")
    parser.add_argument('--experiment_name', type=str, default="NTU_passt_FTtau_441K_FMS_fixh5")
    parser.add_argument('--precision', type=str, default="32")
    parser.add_argument('--ckpt_id', type=str, default="jiw5bohu")
    parser.add_argument('--audio_path', type=str, help="Path to the audio file", default=r"D:\Sean\DCASE\datasets\Extract_to_Folder\TAU-urban-acoustic-scenes-2022-mobile-development\audio\airport-barcelona-0-0-0-a.wav")
    
    # Define model configuration parameters
    parser.add_argument('--n_classes', type=int, default=10)  
    parser.add_argument('--n_mels', type=int, default=128)  
    parser.add_argument('--resample_rate', type=int, default=44100)  
    parser.add_argument('--window_size', type=int, default=800)  
    parser.add_argument('--hop_size', type=int, default=320)  
    parser.add_argument('--n_fft', type=int, default=1024)  
    parser.add_argument('--freqm', type=int, default=48)  
    parser.add_argument('--timem', type=int, default=20)  
    parser.add_argument('--fmin', type=int, default=0)  
    parser.add_argument('--fmax', type=int, default=None)  
    parser.add_argument('--fmin_aug_range', type=int, default=1)  
    parser.add_argument('--fmax_aug_range', type=int, default=1000)  
    parser.add_argument('--arch', type=str, default='passt_s_swa_p16_128_ap476')  
    parser.add_argument('--input_fdim', type=int, default=128)  
    parser.add_argument('--s_patchout_t', type=int, default=0)  
    parser.add_argument('--s_patchout_f', type=int, default=6)  
    
    args = parser.parse_args()

    # Initialize the model from the checkpoint
    assert args.ckpt_id is not None, "A value for argument 'ckpt_id' must be provided."
    ckpt_dir = os.path.join(args.project_name, args.ckpt_id, "checkpoints")
    assert os.path.exists(ckpt_dir), f"No such folder: {ckpt_dir}"
    
    ckpt_file = None
    for file in os.listdir(ckpt_dir):
        if "epoch" in file:
            ckpt_file = os.path.join(ckpt_dir, file)  # choose the best model checkpoint
            print(f"Found checkpoint file: {file}")
            break
    
    assert os.path.exists(ckpt_file), f"No such file: {ckpt_file}. Implement your own mechanism to select the desired checkpoint."
    
    # Load the Lightning module from the checkpoint
    pl_module = PLModule.load_from_checkpoint(ckpt_file, config=args)
    print("Model layers:")
    for name, layer in pl_module.model.named_modules():
        print(f"{name}: {layer}")
    perform_inference_and_visualize(pl_module, args.audio_path)