
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split

import numpy as np
import matplotlib.pyplot as plt
import random
import math
from tqdm import tqdm
from dataset import RadarDatasetClass, collate_fn
from utils import draw_dataset, draw_laser, filter, pc_to_laser, laser_visual_one
from models import RadarTransformer


device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu")

net_transformer = RadarTransformer(
    features=7,
    embed_dim=64,
    nhead=8,
    encoder_layers=6,
    decoder_layers=6,
).to(device)
net_transformer.load_state_dict(torch.load("model/transformer_cgan.pth"))
net_transformer.eval()
l1_loss = nn.L1Loss()

scenes = ['corridor', 'parking']

for scene in scenes:
    dataset = RadarDatasetClass(scene=scene)

    loader = DataLoader(
        dataset=dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    L1 = []
    t = tqdm(loader)
    for l, r in t:
        b_size = l.size(0)

        seq_padded, lens = rnn_utils.pad_packed_sequence(r, batch_first=False)
        max_len = seq_padded.shape[0]
        pad_mask = torch.arange(max_len)[None, :] < lens[:, None]

        seq_padded = seq_padded.to(device)
        pad_mask = ~pad_mask.to(device)

        fake_y = net_transformer(seq_padded, pad_mask).detach().cpu()

        fake_y = torch.clip(fake_y, 0, 5)
        l = torch.clip(l, 0, 5)
        loss_G_l1 = l1_loss(fake_y, l)
        L1.append(loss_G_l1)

    print('scene %s l1 loss: %.4f' % (scene, np.mean(L1)))


'''
cgan transformer accuracy in 100 meter
scene corridor l1 loss: 0.6715
scene parking l1 loss: 1.8636

cgan transformer accuracy in 5 meter
scene corridor l1 loss: 0.1271
scene parking l1 loss: 0.1125
'''
