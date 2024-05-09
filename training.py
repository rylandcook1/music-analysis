import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from dataset import GenreDataset
from neural_net import CNNNetwork

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.00005

# Dataset parameters
ANNOTATIONS_FILE = "shuffled_output.csv"
AUDIO_DIR = "train"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

def create_data_loader(train_data, batch_size):
    """
    Create a DataLoader from the given train_data.

    Args:
    - train_data: The dataset to create DataLoader from.
    - batch_size: Batch size for DataLoader.

    Returns:
    - DataLoader object.
    """
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    """
    Train the model for a single epoch.

    Args:
    - model: The model to train.
    - data_loader: DataLoader containing training data.
    - loss_fn: Loss function to compute loss.
    - optimiser: Optimiser for updating model parameters.
    - device: Device for computation ('cpu' or 'cuda').

    Returns:
    - None
    """
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # Forward pass
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # Backpropagation and optimization
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    """
    Train the model for multiple epochs.

    Args:
    - model: The model to train.
    - data_loader: DataLoader containing training data.
    - loss_fn: Loss function to compute loss.
    - optimiser: Optimiser for updating model parameters.
    - device: Device for computation ('cpu' or 'cuda').
    - epochs: Number of epochs to train.

    Returns:
    - None
    """
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":
    # Check for GPU availability
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    # Instantiate dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    usd = GenreDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)
    train_dataloader = create_data_loader(usd, BATCH_SIZE)

    # Construct model and assign it to device
    cnn = CNNNetwork().to(device)
    print(cnn)

    # Initialise loss function and optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    # Train model
    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # Save trained model
    torch.save(cnn.state_dict(), "BS32E10LR00001.pth")
    print("Trained feed forward net saved at model1.pth")