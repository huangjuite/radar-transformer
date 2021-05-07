
import copy
import math
import random
import numpy as np
from typing import Deque, Dict, List, Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


hyper_parameter = dict(
    kernel=3,
    stride=2,
    padding=2,
    deconv_dim=32,
    deconv_channel=128,
    adjust_linear=235,
    nz=100,
)

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
config = Struct(**hyper_parameter)


# ==========================================================
########################### model ###########################
# ==========================================================


class GeneratorNoise(nn.Module):
    def __init__(self):
        super(GeneratorNoise, self).__init__()
        kernel = 3
        stride = 2
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=kernel, stride=stride),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=kernel, stride=stride),
            nn.ReLU()
        )

        dim = 64*59
        self.linear = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

        self.n_linear = nn.Sequential(
            nn.Linear(config.nz, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.fc_combine = nn.Linear(128*2, 128)

        self.de_fc1 = nn.Sequential(
            nn.Linear(128, config.deconv_channel*config.deconv_dim),
            nn.ReLU()
        )

        self.de_conv = nn.Sequential(
            nn.ConvTranspose1d(config.deconv_channel, config.deconv_channel //
                               2, kernel, stride=stride, padding=config.padding),
            nn.ConvTranspose1d(config.deconv_channel//2, config.deconv_channel //
                               4, kernel, stride=stride, padding=config.padding),
            nn.ConvTranspose1d(config.deconv_channel//4, 1,
                               kernel, stride=stride, padding=config.padding),
        )
        self.adjust_linear = nn.Sequential(
            nn.Linear(config.adjust_linear, 241),
            nn.ReLU()
        )

    def encoder(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def decoder(self, x):
        x = self.de_fc1(x)
        x = x.view(-1, config.deconv_channel, config.deconv_dim)
        x = self.de_conv(x)
        x = self.adjust_linear(x)
        return x

    def forward(self, x, n):
        x = self.encoder(x)
        n = self.n_linear(n)

        x = torch.cat((x, n), dim=-1)
        x = self.fc_combine(x)

        x = self.decoder(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        kernel = 3
        stride = 2
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=kernel, stride=stride),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=kernel, stride=stride),
            nn.ReLU()
        )

        dim = 64*59
        self.linear = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

        self.de_fc1 = nn.Sequential(
            nn.Linear(128, config.deconv_channel*config.deconv_dim),
            nn.ReLU()
        )

        self.de_conv = nn.Sequential(
            nn.ConvTranspose1d(config.deconv_channel, config.deconv_channel //
                               2, kernel, stride=stride, padding=config.padding),
            nn.ConvTranspose1d(config.deconv_channel//2, config.deconv_channel //
                               4, kernel, stride=stride, padding=config.padding),
            nn.ConvTranspose1d(config.deconv_channel//4, 1,
                               kernel, stride=stride, padding=config.padding),
        )
        self.adjust_linear = nn.Sequential(
            nn.Linear(config.adjust_linear, 241),
            nn.ReLU()
        )

    def encoder(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def decoder(self, x):
        x = self.de_fc1(x)
        x = x.view(-1, config.deconv_channel, config.deconv_dim)
        x = self.de_conv(x)
        x = self.adjust_linear(x)
        return x

    def forward(self, x):
        x = self.encoder(x)

        x = self.decoder(x)
        return x

