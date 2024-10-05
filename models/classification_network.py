import torch.nn
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, input_channels, nb_classes):
        super(AlexNet, self).__init__()

        # Extraction caractéristiques
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=0, stride=1)