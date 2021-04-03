
import time
import datetime
import os
import re
import random
import copy
import wandb
import numpy as np
import datetime
from tqdm import tqdm, trange
from typing import Deque, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split

from network_v2 import ActorCNN
from models import RadarEncoder
from dataset import RadarDataset, collate_fn


seed = 888
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

hyper_parameter = dict(
    gamma=0.99,
    lambda_l2=1e-4,       # l2 regularization weight
    epoch=20,
    batch_size=8,
)
wandb.init(config=hyper_parameter,
           project="mmWave-contrastive-transformer", name="cross-attention-mse")

config = wandb.config
print(config)

# dataset
radar_dataset = RadarDataset()

train_loader = DataLoader(
    dataset=radar_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
)


class RDPG(object):

    def __init__(
        self,
        gamma: float = 0.99,
        lambda_l2: float = 1e-4,  # l2 regularization weight
    ):
        """Initialize."""
        obs_dim = 243
        action_dim = 2

        self.gamma = gamma

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # networks CNN
        self.actor_laser = ActorCNN(
            device=self.device,
            in_dim=obs_dim,
            out_dim=action_dim
        ).to(self.device)
        self.actor_laser.load_state_dict(
            torch.load("actor_s1536_f1869509.pth"))
        self.actor_laser.eval()

        # network perceiver
        self.actor_radar = RadarEncoder(
            features=7,
            embed_dim=64,
            nhead=8,
            layers=6,
            output_embeding=512,
        ).to(self.device)

        wandb.watch(self.actor_radar)

        # optimizer
        self.optimizer = optim.Adam(
            self.actor_radar.parameters(),
            lr=3e-4,
            weight_decay=lambda_l2,
        )

    def update_model(self) -> Tuple[torch.Tensor, ...]:
        """Update the model by gradient descent."""
        device = self.device

        t = trange(config.epoch)

        epoch_len = float(len(train_loader))

        for epoch in t:
            i = 0
            for l, r in train_loader:
                b_size = l.size(0)

                seq_padded, lens = rnn_utils.pad_packed_sequence(
                    r, batch_first=False)
                max_len = seq_padded.shape[0]
                pad_mask = torch.arange(max_len)[None, :] < lens[:, None]

                seq_padded = seq_padded.to(device)
                pad_mask = ~pad_mask.to(device)
                l = l.to(device)

                self.optimizer.zero_grad()

                # get latent
                radar_latent = self.actor_radar(seq_padded, pad_mask)
                laser_latent = self.actor_laser(l)

                # mse
                actor_contrastive_loss = torch.mean(
                    (radar_latent-laser_latent).pow(2))

                actor_contrastive_loss.backward()

                self.optimizer.step()

                metrics = {
                    'actor_contrastive_loss': actor_contrastive_loss,
                }
                wandb.log(metrics)

                t.set_description("epoch %.2f" % (i/epoch_len*100))
                i += 1

        # save model
        torch.save(self.actor.state_dict(), os.path.join(
            wandb.run.dir, "model.pth"))


if __name__ == "__main__":
    agent = RDPG(
        gamma=config.gamma,
        lambda_l2=config.lambda_l2,  # l2 regularization weight
    )

    agent.update_model()
