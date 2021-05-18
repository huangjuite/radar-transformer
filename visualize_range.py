
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import RadarDataset
import math

from utils import laser_visual, filter, pc_to_laser, laser_visual_one
from models import RadarTransformer
from network_vae import MMvae
from network_generator import Generator

dataset = RadarDataset()
indx = 84090
l, rp0 = dataset[indx]
_, rp1 = dataset[indx-1]
_, rp2 = dataset[indx-2]
_, rp3 = dataset[indx-3]
_, rp4 = dataset[indx-4]

rp = np.concatenate((rp0, rp1, rp2, rp3, rp4))
rp = filter(rp)

r_laser_np = pc_to_laser(rp[:, 3:6])

lasers = [l, r_laser_np]
show = True
range_limit = 4.9

fig = plt.figure(figsize=(8, 8))
for l in reversed(lasers):
    angle = 120
    xp = []
    yp = []
    for r in l:
        if r <= range_limit:
            yp.append(r * math.cos(math.radians(angle)))
            xp.append(r * math.sin(math.radians(angle)))
        angle -= 1
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.plot(xp, yp, '.')

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.savefig('vis_range.png', bbox_inches='tight', dpi=200)
