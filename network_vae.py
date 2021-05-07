import torch
import torch.nn as nn
import torch.nn.functional as F

hyper_parameter = dict(
    kernel=3,
    stride=2,
    padding=2,
    latent=128,
    deconv_dim=32,
    deconv_channel=128,
    adjust_linear=235,
    epoch=500,
    learning_rate=0.001,
)
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
config = Struct(**hyper_parameter)

class MMvae(nn.Module):
    def __init__(self):
        super(MMvae, self).__init__()
        kernel = 3
        stride = 2
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=kernel, stride=stride),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=kernel, stride=stride),
            nn.ReLU()
        )

        dim = 64*59
        self.linear1 = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU()
        )
        self.en_fc1 = nn.Linear(512, config.latent)
        self.en_fc2 = nn.Linear(512, config.latent)

        self.de_fc1 = nn.Sequential(
            nn.Linear(config.latent, config.deconv_channel*config.deconv_dim),
            nn.ReLU()
        )

        self.de_conv = nn.Sequential(
            nn.ConvTranspose1d(config.deconv_channel, config.deconv_channel //
                               2, kernel, stride=stride, padding=config.padding),
            # nn.ReLU(),
            nn.ConvTranspose1d(config.deconv_channel//2, config.deconv_channel //
                               4, kernel, stride=stride, padding=config.padding),
            # nn.ReLU(),
            nn.ConvTranspose1d(config.deconv_channel//4, 1,
                               kernel, stride=stride, padding=config.padding),
            # nn.ReLU(),
        )
        self.adjust_linear = nn.Sequential(
            nn.Linear(config.adjust_linear, 241),
            nn.ReLU()
        )

    def encoder(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        mean = self.en_fc1(x)
        logvar = self.en_fc2(x)
        return mean, logvar

    def reparameter(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean + eps*std

    def decoder(self, x):
        x = self.de_fc1(x)
        x = x.view(-1, config.deconv_channel, config.deconv_dim)
        x = self.de_conv(x)
        x = self.adjust_linear(x)
        return x

    def forward(self, x):
        mean, logvar = self.encoder(x)
        x = self.reparameter(mean, logvar)
        x = self.decoder(x)
        return x, mean, logvar
