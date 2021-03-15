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
    batch_size=8,
    nhead_attention=8,
    encoder_layer=6,
    decoder_layer=6,
    feature_dim=7,
    embed_dim=64,
    split_ratio=0.8,
    epoch=25,
    beta1=0.5,
    learning_rate=0.0002,
    lambda_l1=1,
    vis_num=4,
    visualize_epoch=2,

)

wandb.init(config=hyper_parameter,
           project="radar-transformer", name='test')
config = wandb.config


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(device)

# dataset
radar_dataset = RadarDataset()

train_len = int(config.split_ratio*len(radar_dataset))
test_len = len(radar_dataset) - train_len

train_dataset, test_dataset = random_split(
    radar_dataset, [train_len, test_len])

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
)


# pytorch_total_params = sum(p.numel() for p in model.parameters())
# print(pytorch_total_params)


# for l, r in train_loader:
#     seq_padded, lens = rnn_utils.pad_packed_sequence(r, batch_first=False)
#     max_len = seq_padded.shape[0]
#     pad_mask = torch.arange(max_len)[None, :] < lens[:, None]

#     seq_padded = seq_padded.to(device)
#     pad_mask = ~pad_mask.to(device)

#     y = model(seq_padded, pad_mask)
#     # print(y.shape)
#     y = y.detach()
#     # break


netG = RadarTransformer(
    features=config.feature_dim,
    embed_dim=config.embed_dim,
    nhead=config.nhead_attention,
    encoder_layers=config.encoder_layer,
    decoder_layers=config.decoder_layer,
).to(device)

# netD = Discriminator().to(device)
netD = DiscriminatorTransform(
    features=config.feature_dim,
    embed_dim=config.embed_dim,
    nhead=config.nhead_attention,
    encoder_layers=config.encoder_layer,
).to(device)

wandb.watch(netG)
wandb.watch(netD)

# optimizers
optimizer_g = optim.Adam(netG.parameters(),
                         lr=config.learning_rate, betas=(config.beta1, 0.999))
optimizer_d = optim.Adam(netD.parameters(),
                         lr=config.learning_rate, betas=(config.beta1, 0.999))

# criterion
# gan_loss = nn.BCEWithLogitsLoss()
gan_loss = nn.MSELoss()
l1_loss = nn.L1Loss()


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

        fake_y = netG(seq_padded, pad_mask)
        fake_y = torch.unsqueeze(fake_y, 1)

        y = l.to(device)
        y = torch.unsqueeze(y, 1)

        # patch size 14
        fake_label = Variable(torch.Tensor(
            np.zeros((b_size, 14))), requires_grad=False).to(device)
        real_label = Variable(torch.Tensor(
            np.ones((b_size, 14))), requires_grad=False).to(device)

        ########################### train D ############################

        set_requires_grad(netD, True)
        optimizer_d.zero_grad()

        # fake
        pred_fake = netD(seq_padded, pad_mask, fake_y.detach())
        loss_D_fake = gan_loss(pred_fake, fake_label)

        # real
        pred_real = netD(seq_padded, pad_mask, y)
        loss_D_real = gan_loss(pred_real, real_label)

        # train
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        optimizer_d.step()

        ########################### train G ############################

        set_requires_grad(netD, False)
        optimizer_g.zero_grad()

        pred_fake = netD(seq_padded, pad_mask, fake_y)
        loss_G_gan = gan_loss(pred_fake, real_label)
        loss_G_l1 = l1_loss(fake_y, y) * config.lambda_l1
        loss_G = loss_G_gan + loss_G_l1
        loss_G.backward()
        optimizer_g.step()

        ########################### log ##################################

        metrics = {
            'loss_D_real': loss_D_real,
            'loss_D_fake': loss_D_fake,
            'loss_D': loss_D,
            'loss_G_gan': loss_G_gan,
            'loss_G_l1': loss_G_l1,
            'loss_G': loss_G,
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
