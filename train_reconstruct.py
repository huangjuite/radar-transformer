import os
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split

from dataset import RadarDataset, collate_fn
from models import RadarTransformer, DiscriminatorTransform
from utils import *

torch.random.manual_seed(777)

hyper_parameter = dict(
    batch_size=24,
    nhead_attention=8,
    encoder_layer=6,
    decoder_layer=6,
    feature_dim=7,
    embed_dim=64,
    epoch=25,
    beta1=0.5,
    learning_rate=3e-4,
    vis_num=4,
    visualize_epoch=2,
)

wandb.init(config=hyper_parameter,
           project="radar-transformer", name='generator-only')
config = wandb.config


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(device)

# dataset
radar_dataset = RadarDataset()

train_loader = DataLoader(
    dataset=radar_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
)


# pytorch_total_params = sum(p.numel() for p in model.parameters())
# print(pytorch_total_params)

netG = RadarTransformer(
    features=config.feature_dim,
    embed_dim=config.embed_dim,
    nhead=config.nhead_attention,
    encoder_layers=config.encoder_layer,
    decoder_layers=config.decoder_layer,
).to(device)

wandb.watch(netG)

# optimizers
optimizer_g = optim.Adam(netG.parameters(),
                         lr=config.learning_rate, betas=(config.beta1, 0.999))

# criterion
# gan_loss = nn.BCEWithLogitsLoss()
criterion = nn.l1Loss()


def set_requires_grad(net, requires_grad=False):
    for param in net.parameters():
        param.requires_grad = requires_grad


step = 0
t = trange(config.epoch)

for epoch in t:
    for l, r in train_loader:
        b_size = l.size(0)

        seq_padded, lens = rnn_utils.pad_packed_sequence(r, batch_first=False)
        max_len = seq_padded.shape[0]
        pad_mask = torch.arange(max_len)[None, :] < lens[:, None]

        seq_padded = seq_padded.to(device)
        pad_mask = ~pad_mask.to(device)

        optimizer_g.zero_grad()

        fake_y = netG(seq_padded, pad_mask)

        y = l.to(device)


        ########################### train G ############################
        
        loss = criterion(fake_y, l)

        loss.backward()

        optimizer_g.step()

        ########################### log ##################################

        metrics = {
            'loss': loss,
        }
        wandb.log(metrics)
        step += 1
        t.set_description('setp: %d' % step)

        ########################### visualize ##################################

        if (step / len(train_loader)) % config.visualize_epoch == 0:

            fake_y = fake_y.detach().cpu().numpy(
            ).reshape(-1, 241)[:config.vis_num]
            l = l.detach().cpu().numpy().reshape(-1, 241)[:config.vis_num]

            examples = []
            for i, (y, laser) in enumerate(zip(fake_y, l)):
                examples.append(laser_visual([y, laser], show=False))

            wandb.log({"example_%d" %
                       step: [wandb.Image(img) for img in examples]})


################## save model #################
torch.save(netG.state_dict(), os.path.join(
    wandb.run.dir, "model.pth"))
