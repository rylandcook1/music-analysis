from pathlib import Path
import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

genres = [
    "country",
    "pop",
    "rock",
    "jazz",
    "metal"
]


def load_audio(audio_path, target_rate, mono):
    initial_signal, sample_rate = librosa.load(audio_path,
                                               sr=target_rate,
                                               mono=mono)
    tensor_signal = torch.from_numpy(initial_signal)
    tensor_signal.unsqueeze_(0)
    return tensor_signal, sample_rate


class CustomSongDataset(Dataset):

    # annotation --> contains information about class, filename, etc.
    # audio_directory --> directory containing all the audio files
    def __init__(self,
                 annotation_file,
                 audio_directory,
                 transformation,
                 target_sample_rate,
                 num_samples
                 ):
        self.annotation = pd.read_csv(annotation_file)
        self.audio_directory = audio_directory
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        audio_sample_path = self.get_audio_sample_path(index)
        label = self.get_audio_sample_label(index)
        # read the audio signal in from librosa, then transform to the proper tensor form using helper function
        mySignal, sample_rate = load_audio(audio_sample_path,
                                           self.target_sample_rate,
                                           True)
        # we have generated a signal with 1 channel at the target_sample_rate specified in constructor
        # now, we are going to transform the data into a melspectrogram
        mySignal = self.cut_if_needed(mySignal)
        mySignal = self.pad_if_needed(mySignal)
        mySignal = self.transformation(mySignal)
        return mySignal, label

    def _get_audio_sample_path(self, index):
        path = os.path.join(self.audio_directory, self.annotation.iloc[index, 0])
        return path

    def get_audio_sample_label(self, index):
        return self.annotation.iloc[index, 2]  # class label name, from annotation

    def cut_if_needed(self, mySignal):
        if mySignal.shape[1] > self.num_samples:
            mySignal = mySignal[:self.num_samples]
        return mySignal

    def pad_if_needed(self, mySignal):
        sig_length = mySignal.shape[1]
        if sig_length < self.num_samples:
            num_missing_samples = self.num_samples - sig_length
            last_dim_padding = (0, num_missing_samples)
            mySignal = torch.nn.functional.pad(mySignal, last_dim_padding)
        return mySignal

    def plot_waveform(self, index, xlim=None, ylim=None):
        audio_sample_path = self._get_audio_sample_path(index)
        waveform, sample_rate = load_audio(audio_sample_path,
                                           self.target_sample_rate,
                                           True)

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sample_rate

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].plot(time_axis, waveform[c], linewidth=1)
            axes[c].grid(True)
            if num_channels > 1:
                axes[c].set_ylabel(f'Channel {c + 1}')
            if xlim:
                axes[c].set_xlim(xlim)
            if ylim:
                axes[c].set_ylim(ylim)
        figure.suptitle(f"Sample {index} Waveform")
        plt.show()

    def plot_melspectrogram(self, index, ylabel="freq_bin", ax=None):
        audio_sample_path = self._get_audio_sample_path(index)
        initial, sr = librosa.load(audio_sample_path,
                                   sr=self.target_sample_rate,
                                   mono=True)
        S = librosa.feature.melspectrogram(y=initial,
                                           sr=sr,
                                           n_fft=1024,
                                           hop_length=512,
                                           n_mels=64)
        S_dB = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr)
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Sample {index} Mel Spectrogram")
        plt.show()


if __name__ == "__main__":
    ANNOTATION_FILE = Path.cwd().joinpath("test_train.csv")  # path of annotation file, in the working directory
    AUDIO_DIRECTORY = Path.cwd().joinpath("train")  # path of audio directory, in the working directory
    SAMPLE_RATE = 16000  # default sample rate
    NUM_SAMPLES = 240000  # 15 seconds at the SAMPLE_RATE constant

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    songs_dataset = CustomSongDataset(ANNOTATION_FILE,
                                      AUDIO_DIRECTORY,
                                      mel_spectrogram,
                                      SAMPLE_RATE,
                                      NUM_SAMPLES)
    print(f"There are {len(songs_dataset)} samples in the dataset.")
    songs_dataset.plot_waveform(24)
    songs_dataset.plot_melspectrogram(24)
