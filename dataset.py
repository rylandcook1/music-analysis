import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os

class GenreDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, num_samples, device):
        # Initialize the GenreDataset class
        self.annotations = pd.read_csv(annotations_file)  # Read annotations CSV file
        self.audio_dir = audio_dir  # Directory containing audio samples
        self.transformation = transformation  # Audio transformation (e.g., Mel spectrogram)
        self.target_sample_rate = target_sample_rate  # Target sample rate for audio
        self.num_samples = num_samples  # Number of audio samples
        self.device = device  # Device for computation (e.g., 'cpu' or 'cuda')

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.annotations)

    def __getitem__(self, index):
        # Get audio sample and label for a given index
        audio_sample_path = self._get_audio_sample_path(index)  # Get audio sample path
        label = self._get_audio_sample_label(index)  # Get audio sample label
        signal, sr = torchaudio.load(audio_sample_path)  # Load audio signal and sample rate
        signal = self._resample_if_necessary(signal, sr)  # Resample audio signal if necessary
        signal = self._mix_down_if_necessary(signal)  # Mix down audio signal if necessary
        signal = self.transformation(signal)  # Apply audio transformation
        return signal, label

    def _resample_if_necessary(self, signal, sr):
        # Resample audio signal if sample rate is different from target sample rate
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        # Mix down audio signal if it has multiple channels
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        # Get the file path of an audio sample for a given index
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 0])
        return path

    def _get_audio_sample_label(self, index):
        # Get the label of an audio sample for a given index
        return self.annotations.iloc[index, 2]

if __name__ == "__main__":
    # Constants
    ANNOTATIONS_FILE = "music-analysis-annotation.csv"
    AUDIO_DIR = "yes"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    device = "cpu"  # Device for computation

    # Mel spectrogram transformation
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    # Create an instance of GenreDataset
    usd = GenreDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)

    # Print the number of samples in the dataset
    print(f"There are {len(usd)} samples")

    # Get the first sample and its label
    signal, label = usd[0]
    print(signal, label)