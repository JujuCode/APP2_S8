import torch
import torch.nn as nn

class YOLO(nn.Module):
    def __init__(self, input_channels):
        super(YOLO, self).__init__()

        # Fonction ReLU
        self.leakyRelu = nn.LeakyReLU()


        # Séquence YOLO
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1, stride=1)
        # LeakyRelu
        self.maxPool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        # LeakyRelu
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0, stride=2)
        # LeakyRelu

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=100, kernel_size=3, padding=0, stride=1)
        # LeakyRelu
        self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv5 = nn.Conv2d(in_channels=100, out_channels=220, kernel_size=2, padding=0, stride=1)
        self.relu = nn.ReLU()
        self.conv6 = nn.Conv2d(in_channels=220, out_channels=882, kernel_size=1, padding=0, stride=1)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        # Conv 1
        x = self.conv1(x)
        x = self.leakyRelu(x)
        x = self.maxPool1(x)

        # Conv 2
        x = self.conv2(x)
        x = self.leakyRelu(x)
        x = self.maxPool2(x)

        # Conv 3
        x = self.conv3(x)
        x = self.leakyRelu(x)

        # Conv 4
        x = self.conv4(x)
        x = self.leakyRelu(x)
        x = self.maxPool3(x)

        # Conv 5 et 6
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)

        # Reshape
        x = torch.reshape(x, (32, 18, 7, 7))

        # Sigmoid
        x = self.sigmoid(x)

        return x