from pathlib import Path
import librosa
import pandas as pd
import torch
import torchaudio
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

    def get_audio_sample_path(self, index):
        folder = "fold" + str(self.annotation.iloc[index, 1])  # grabs the name of the folder from the annotation
        path = Path.joinpath(self.audio_directory, folder, self.annotation.iloc[index, 0])
        # pieces the filepath
        # together with the directory, along with the folder and filename from the annotation
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


if __name__ == "__main__":
    ANNOTATION_FILE = Path.cwd().joinpath("annotation.csv")  # path of annotation file, in the working directory
    AUDIO_DIRECTORY = Path.cwd().joinpath("audio")  # path of audio directory, in the working directory
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
    signal, label = songs_dataset[300]
    print(signal)
