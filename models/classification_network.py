import torch.nn
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, input_channels, nb_classes):
        super(AlexNet, self).__init__()

        # Extraction caractéristiques (Starting size: 53 x 53)
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=0, stride=1)  # Output size: 51 x 51
        self.ReLU = nn.ReLU()
        self.maxPool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)  # Output size: 25 x 25
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0, stride=2)  # Output size: 12 x 12
        self.maxPool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)  # Output size: 5 x 5
        self.flatten = nn.Flatten()  # Reel = 3200

        # Classification
        self.linear1 = nn.Linear(in_features=3200, out_features=39)
        self.linear2 = nn.Linear(in_features=39, out_features=10)
        self.linear3 = nn.Linear(in_features=10, out_features=nb_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.ReLU(x)
        x = self.maxPool1(x)

        x = self.conv2(x)
        x = self.ReLU(x)
        x = self.maxPool2(x)

        x = self.flatten(x)

        x = self.linear1(x)
        x = self.ReLU(x)
        x = self.linear2(x)
        x = self.ReLU(x)
        x = self.linear3(x)
        x = self.sigmoid(x)

        return x



