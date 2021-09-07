import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchsummary import summary



class encoder(nn.Module):
    def __init__(self, latent_size, lr):
        super(encoder, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc = nn.Linear(262144, 100)

    def forward(self, images):
        image = images.view(-1, 1, 32, 32)
        convs = self.seq(image)
        output = self.fc(convs)
        return output

