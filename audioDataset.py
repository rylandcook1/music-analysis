import numpy as np
from pathlib import Path
import torch
import librosa
from torch.utils.data import Dataset
import pandas as pd
import torchaudio


class CustomSongDataset(Dataset):

    # annotation --> contains information about class, filename, etc.
    # audio_directory --> directory containing all the audio files
    def __init__(self,
                 annotation_file,
                 audio_directory,
                 transformation,
                 target_sample_rate,
                 num_samples):
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
        mySignal, sample_rate = librosa.load(audio_sample_path,
                                             sr=self.target_sample_rate,
                                             mono=True)
        # we have generated a signal with 1 channel at the target_sample_rate
        # specified in the constructor
        # now, we are going to transform the data into a melspectrogram
        mySignal = self.cut_if_needed(mySignal)
        mySignal = self.pad_if_needed(mySignal)
        mySignal = self.transformation(mySignal, sample_rate)
        return mySignal, label

    def get_audio_sample_path(self, index):
        folder = "fold" + str(self.annotation.iloc[index, 1])  # grabs the name of the folder from the annotation
        path = Path.joinpath(self.audio_directory, folder, self.annotation.iloc[index, 0])
        # pieces the filepath
        # together with the directory, along with the folder and filename from the annotation
        return path

    def get_audio_sample_label(self, index):
        return self.annotation.iloc[index, 3]  # class label name, from annotation

    def cut_if_needed(self, mySignal):
        if mySignal.shape[0] > self.num_samples:
            mySignal = mySignal[:self.num_samples]
        return mySignal

    def pad_if_needed(self, mySignal):
        sig_length = mySignal.shape[0]
        if sig_length < self.num_samples:
            num_missing_samples = self.num_samples - sig_length
            last_dim_padding = (0, num_missing_samples)
            tensor = torch.from_numpy(mySignal)
            tensor = torch.nn.functional.pad(tensor, last_dim_padding)
            mySignal = np.array(tensor)
        return mySignal


ANNOTATION_FILE = Path.cwd().joinpath("annotation.csv")  # path of annotation file, in the working directory
AUDIO_DIRECTORY = Path.cwd().joinpath("audio")  # path of audio directory, in the working directory
SAMPLE_RATE = 16000  # default sample rate
NUM_SAMPLES = 245000  # 15 seconds at the SAMPLE_RATE constant


def mel_spectrogram(signal, sample_rate):
    return librosa.feature.melspectrogram(y=signal,
                                          sr=sample_rate,
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
signal, label = songs_dataset[300]
print(signal)
