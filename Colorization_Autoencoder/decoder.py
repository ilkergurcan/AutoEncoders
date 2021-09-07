import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchsummary import summary

class decoder(nn.Module):
    def __init__(self, latent_size, lr):
        super(decoder, self).__init__()

        self.fc = nn.Linear(latent_size, 256*32*32)

        self.seq = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        hidden = self.fc(z)
        flatten = hidden.view(-1, 256, 32, 32)
        seq = self.seq(flatten)
        return seq