import torch.nn
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, input_channels, nb_classes):
        super(UNet, self).__init__()

        # Fonction ReLU et sigmoid
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


        # Down 1
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=28, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(28)
        self.conv1_2 = nn.Conv2d(in_channels=28, out_channels=28, kernel_size=3, padding=1, stride=1)
        self.bn1_2 = nn.BatchNorm2d(28)
        self.maxPool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Down 2
        self.conv2 = nn.Conv2d(in_channels=28, out_channels=56, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(56)
        self.conv2_2 = nn.Conv2d(in_channels=56, out_channels=56, kernel_size=3, padding=1, stride=1)
        self.bn2_2 = nn.BatchNorm2d(56)
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Down 3
        self.conv3 = nn.Conv2d(in_channels=56, out_channels=112, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(112)
        self.conv3_2 = nn.Conv2d(in_channels=112, out_channels=112, kernel_size=3, padding=1, stride=1)
        self.bn3_2 = nn.BatchNorm2d(112)
        self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Down 4
        self.conv4 = nn.Conv2d(in_channels=112, out_channels=224, kernel_size=3, padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(224)
        self.conv5 = nn.Conv2d(in_channels=224, out_channels=112, kernel_size=3, padding=1, stride=1)
        self.bn5 = nn.BatchNorm2d(112)

        # UP 6
        self.convT6 = nn.ConvTranspose2d(in_channels=112, out_channels=112, kernel_size=3, padding=0, stride=2)
        self.bnT6 = nn.BatchNorm2d(112)
        self.conv6 = nn.Conv2d(in_channels=224, out_channels=56, kernel_size=3, padding=1, stride=1)
        self.bn6 = nn.BatchNorm2d(56)

        # UP 7
        self.convT7 = nn.ConvTranspose2d(in_channels=56, out_channels=56, kernel_size=2, padding=0, stride=2)
        self.bnT7 = nn.BatchNorm2d(56)
        self.conv7 = nn.Conv2d(in_channels=112, out_channels=28, kernel_size=3, padding=1, stride=1)
        self.bn7 = nn.BatchNorm2d(28)

        # UP 8
        self.convT8 = nn.ConvTranspose2d(in_channels=28, out_channels=28, kernel_size=3, padding=0, stride=2)
        self.bnT8 = nn.BatchNorm2d(28)
        self.conv8 = nn.Conv2d(in_channels=56, out_channels=14, kernel_size=3, padding=1, stride=1)
        self.bn8 = nn.BatchNorm2d(14)
        self.conv9 = nn.Conv2d(in_channels=14, out_channels=nb_classes, kernel_size=1, padding=0, stride=1)

    def forward(self, x):

        # Down 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu(x)
        after_relu_1 = x
        x = self.maxPool1(x)

        # Down 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu(x)
        after_relu_2 = x
        x = self.maxPool2(x)

        # Down 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.relu(x)
        after_relu_3 = x
        x = self.maxPool3(x)

        # Down 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        # UP 6
        x = self.convT6(x)
        x = self.bnT6(x)
        x = torch.cat((x, after_relu_3), dim=1)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)

        # UP 7
        x = self.convT7(x)
        x = self.bnT7(x)
        x = torch.cat((x, after_relu_2), dim=1)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu(x)

        # UP 8
        x = self.convT8(x)
        x = self.bnT8(x)
        x = torch.cat((x, after_relu_1), dim=1)
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)
        x = self.conv9(x)
        x = self.sigmoid(x)

        return x
