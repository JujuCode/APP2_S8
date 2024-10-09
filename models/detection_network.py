import torch
import torch.nn as nn

# class YOLO(nn.Module):
#     def __init__(self, input_channels):
#         super(YOLO, self).__init__()
#
#         # Fonction ReLU
#         self.leakyRelu = nn.LeakyReLU()
#
#
#         # Séquence YOLO
#         self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1, stride=1)
#         # LeakyRelu
#         self.maxPool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
#         # LeakyRelu
#         self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0, stride=2)
#         # LeakyRelu
#
#         self.conv4 = nn.Conv2d(in_channels=64, out_channels=100, kernel_size=3, padding=0, stride=1)
#         # LeakyRelu
#         self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#
#         self.conv5 = nn.Conv2d(in_channels=100, out_channels=220, kernel_size=2, padding=0, stride=1)
#         self.relu = nn.ReLU()
#         self.conv6 = nn.Conv2d(in_channels=220, out_channels=882, kernel_size=1, padding=0, stride=1)
#
#         self.sigmoid = nn.Sigmoid()
#
#
#     def forward(self, x):
#
#         # Conv 1
#         x = self.conv1(x)
#         x = self.leakyRelu(x)
#         x = self.maxPool1(x)
#
#         # Conv 2
#         x = self.conv2(x)
#         x = self.leakyRelu(x)
#         x = self.maxPool2(x)
#
#         # Conv 3
#         x = self.conv3(x)
#         x = self.leakyRelu(x)
#
#         # Conv 4
#         x = self.conv4(x)
#         x = self.leakyRelu(x)
#         x = self.maxPool3(x)
#
#         # Conv 5 et 6
#         x = self.conv5(x)
#         x = self.relu(x)
#         x = self.conv6(x)
#
#         # # Reshape
#         x = torch.reshape(x, (32, 18, 7, 7))
#
#         # Sigmoid
#         x = self.sigmoid(x)
#
#         return x

class RCNN(nn.Module):
    def __init__(self, input_channels):
        super(RCNN, self).__init__()

        # Fonction ReLU
        self.relu = nn.ReLU()

        # Séquence RCNN
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=5, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=3, padding=0)
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(in_features=4096, out_features=80)
        self.linear2 = nn.Linear(in_features=80, out_features=64)
        self.linear3 = nn.Linear(in_features=64, out_features=21)

        # Fonction softmax
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        # Fonction batch norm
        self.batchnorm1 = nn.BatchNorm2d(num_features=16)
        self.batchnorm2 = nn.BatchNorm2d(num_features=32)
        self.batchnorm3 = nn.BatchNorm2d(num_features=64)
        self.batchnorm4 = nn.BatchNorm2d(num_features=64)

    def forward(self, x):

        # Conv 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)


        # Conv 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)

        # Maxpool 1
        x = self.maxpool1(x)

        # Conv 3
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)

        # Conv 4
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.relu(x)

        # Max pool 2
        x = self.maxpool2(x)

        # Flatten
        x = self.flatten(x)

        # Linear 1
        x = self.linear1(x)
        x = self.relu(x)
        # x = self.dropout(x)

        # Linear 2
        x = self.linear2(x)
        x = self.relu(x)

        # Linear 3
        x = self.linear3(x)
        # x = self.relu(x)

        x = self.sigmoid(x)

        # Reshape
        x = x.view(-1, 3, 7)

        return x
