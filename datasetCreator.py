import os
from torch.utils.data import Dataset
import pandas as pd
import torchaudio


class CustomSongDataset(Dataset):

    # annotation --> contains information about class, filename, etc.
    # audio_directory --> directory containing all the audio files
    def __init__(self, annotation_file, audio_directory):
        self.annotation = pd.read_csv(annotation_file)
        self.audio_directory = audio_directory

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sample_rate = torchaudio.load(audio_sample_path)
        return signal, label

    def _get_audio_sample_path(self, index):
        folder = f"fold{self.annotation.iloc[index, 0]}"  # TODO: update to proper value of folder
        path = os.path.join(self.audio_directory, folder, self.annotation.iloc[index, 0])  # TODO: update proper value of filename
        return path

    def _get_audio_sample_label(self, index):
        return self.annotation.iloc[index, 0]  # TODO: update to proper value of label

