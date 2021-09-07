import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchsummary import summary
import torch as T
import os

class autoencoder(nn.Module):
    def __init__(self, latent_size):
        super(autoencoder, self).__init__()

        self.checkpoint_dir = "models/"
        self.checkpoint_file = os.path.join(self.checkpoint_dir, "autoencoder")

        self.encoder_seq = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Flatten()
        )

        self.encoder_fc = nn.Linear(262144, 100)

        self.decoder_fc = nn.Linear(latent_size, 256 * 32 * 32)

        self.decoder_seq = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.optimizer = optim.Adam(self.parameters())
        self.loss = nn.MSELoss()

    def forward(self, images):
        #encoder
        image = images.view(-1, 1, 32, 32)
        convs = self.encoder_seq(image)
        z = self.encoder_fc(convs)

        #decoder
        hidden = self.decoder_fc(z)
        flatten = hidden.view(-1, 256, 32, 32)
        colorized = self.decoder_seq(flatten)

        return colorized

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))