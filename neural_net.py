from torch import nn
from torchsummary import summary

class CNNNetwork(nn.Module):
    def __init__(self):
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
        self.linear = nn.Linear(26880, 5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        X = self.conv1(input_data)
        X = self.conv2(X)
        X = self.conv3(X)
        X = self.conv4(X)
        X = self.flatten(X)
        logits = self.linear(X)
        #predictions = self.softmax(logits)
        return logits

if __name__ == "__main__":
    cnn = CNNNetwork()
    summary(cnn, (1, 64, 44))