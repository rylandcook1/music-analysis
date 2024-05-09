from torch import nn
from torchsummary import summary

class CNNNetwork(nn.Module):
    def __init__(self):
        """
        Initialize the CNNNetwork class.

        The network consists of 4 convolutional blocks followed by a flatten layer,
        and a fully connected linear layer for classification.
        """
        super().__init__()
        # 4 convolutional blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(26880, 5)  # 26880 = 128 * 5 * 44 (output size after max pooling)

    def forward(self, input_data):
        """
        Forward pass through the CNNNetwork.

        Args:
        - input_data: Input data tensor.

        Returns:
        - logits: Output logits tensor.
        """
        X = self.conv1(input_data)
        X = self.conv2(X)
        X = self.conv3(X)
        X = self.conv4(X)
        X = self.flatten(X)
        logits = self.linear(X)
        return logits

if __name__ == "__main__":
    # Create an instance of the CNNNetwork
    cnn = CNNNetwork()

    # Print model summary
    summary(cnn, (1, 64, 44))  # Input shape: (channels, height, width)