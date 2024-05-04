import torch
import torchaudio
from torch import nn
from pathlib import Path
from torch.utils.data import DataLoader
from audioDataset import CustomSongDataset
from neuralNetwork import CNNNetwork

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
ANNOTATION_FILE = Path.cwd().joinpath("annotation.csv")  # path of annotation file, in the working directory
AUDIO_DIRECTORY = Path.cwd().joinpath("audio")  # path of audio directory, in the working directory
SAMPLE_RATE = 16000  # default sample rate
NUM_SAMPLES = 240000  # 15 seconds at the SAMPLE_RATE constant


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_epoch(model, data_loader, loss_fn, optimiser):
    for input, target in data_loader:
        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)
        # backpropogation and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_epoch(model, data_loader, loss_fn, optimiser)
        print("_______________________________")
    print("Finished training!")


if __name__ == "__main__":
    # make the melspectrogram transform function
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )


    # instantiate dataset object and create data loader
    songs_dataset = CustomSongDataset(ANNOTATION_FILE,
                                      AUDIO_DIRECTORY,
                                      mel_spectrogram,
                                      SAMPLE_RATE,
                                      NUM_SAMPLES)
    train_dataloader = create_data_loader(songs_dataset, BATCH_SIZE)

    cnn = CNNNetwork()
    print(cnn)

    # initialise loss (cost) function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimiser, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), "cnn_audio.pth")
    print("Trained network saved at cnn_audio.pth")
