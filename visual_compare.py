
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import RadarDataset


from utils import draw_dataset, draw_laser, filter, pc_to_laser, laser_visual_one
from models import RadarTransformer
from network_vae import MMvae
from network_generator import Generator

dataset = RadarDataset()

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu")

net_cgan = Generator().to(device)
net_cgan.eval()
net_cgan.load_state_dict(torch.load("model/cgan/0827_1851.pth"))

net_vae = MMvae().to(device)
net_vae.eval()
net_vae.load_state_dict(torch.load("model/vae/0726_1557.pth"))

net_transformer = RadarTransformer(
    features=7,
    embed_dim=64,
    nhead=8,
    encoder_layers=6,
    decoder_layers=6,
).to(device)
net_transformer.load_state_dict(torch.load("model/transformer_cgan.pth"))
net_transformer.eval()


def reconstruct(indx=39714):
    l, rp0 = dataset[indx]
    _, rp1 = dataset[indx-1]
    _, rp2 = dataset[indx-2]
    _, rp3 = dataset[indx-3]
    _, rp4 = dataset[indx-4]

    rp = np.concatenate((rp0, rp1, rp2, rp3, rp4))
    rp = filter(rp)

    r0_pc = rp0[:, 3:6]
    r_laser_np = pc_to_laser(rp[:, 3:6])
    # draw_dataset(l, rp)
    # draw_laser(l, r_laser_np)

    r_laser = torch.Tensor(r_laser_np).reshape(1, 1, -1).to(device)

    recon_cgan = net_cgan(r_laser).detach().cpu().numpy()[0]
    recon_cgan = np.clip(recon_cgan, 0, 5)[0]
    recon_vae, _, _ = net_vae(r_laser)
    recon_vae = recon_vae.detach().cpu().numpy()[0]
    recon_vae = np.clip(recon_vae, 0, 5)[0]

    r_t = torch.Tensor(rp0).to(device)
    r_t = torch.unsqueeze(r_t, dim=1)
    recon_transformer = net_transformer(r_t, None)
    recon_transformer = recon_transformer.detach().cpu().numpy()

    return l, r_laser_np, r0_pc, recon_cgan, recon_vae, recon_transformer


indexs = [110379, 81471, 74911, 44003]

f_size = 18
marker_size = 2
fig_names = ['mmWave', 'mmWave\nfiltered',
             'VAE', 'cGAN', 'cGAN\ntransformer', 'LiDAR']

fig, axs = plt.subplots(len(fig_names), len(indexs))
fig.set_figheight(16)
fig.set_figwidth(12)

for i, indx in enumerate(indexs):
    l, r_laser_np, r0_pc, recon_cgan, recon_vae, recon_transformer = reconstruct(
        indx)
    r0_laser = pc_to_laser(r0_pc)
    laser_visual_one(r0_laser, axs[0, i], m_size=marker_size)
    laser_visual_one(r_laser_np, axs[1, i], m_size=marker_size)
    laser_visual_one(recon_vae, axs[2, i], m_size=marker_size)
    laser_visual_one(recon_cgan, axs[3, i], m_size=marker_size)
    laser_visual_one(recon_transformer, axs[4, i], m_size=marker_size)
    laser_visual_one(l, axs[5, i], m_size=marker_size)

places = ['corridor', 'narrow\npassage', 'corner', 'parking']
for i in range(len(places)):
    axs[0, i].set_title(places[i], fontsize=f_size)

for i, ax in enumerate(axs.flat):
    ax.set_xlabel('meter', fontsize=f_size)
    ax.set_ylabel('%s' % (fig_names[i//len(places)]),
                  fontsize=f_size, rotation=0)
    ax.yaxis.set_label_coords(-0.6, 0.5)

for ax in axs.flat:
    ax.label_outer()

plt.gcf().subplots_adjust(left=0.18)

# plt.show()
plt.savefig('visual_compare.png',bbox_inches = 'tight', dpi=200)


#####################
# plot 3D
#####################
'''
indexs = [110379, 39714, 74911, 43030]

f_size = 18
marker_size = 2
fig_names = ['mmWave', 'mmWave\nfiltered',
             'VAE\ncnn', 'cGAN\ncnn', 'cGAN\ntransformer', 'LiDAR']

fig = plt.figure(figsize=(10, 16))
step = 1
nsample = 1
axs = [[],[],[],[],[],[]]
axsflat = []
for i, indx in enumerate(indexs):
    l, r_laser_np, r0_pc, recon_cgan, recon_vae, recon_transformer = reconstruct(
        indx)

    ax = fig.add_subplot(len(fig_names), len(indexs), nsample, projection='3d')
    ax.scatter3D(r0_pc[:, 0], r0_pc[:, 1], r0_pc[:, 2])
    axs[0].append(ax)
    axsflat.append(ax)

    ax = fig.add_subplot(len(fig_names), len(indexs), nsample+4)
    laser_visual_one(r_laser_np, ax, m_size=marker_size)
    axs[1].append(ax)
    axsflat.append(ax)

    ax = fig.add_subplot(len(fig_names), len(indexs), nsample+8)
    laser_visual_one(recon_vae, ax, m_size=marker_size)
    axs[2].append(ax)
    axsflat.append(ax)

    ax = fig.add_subplot(len(fig_names), len(indexs), nsample+12)
    laser_visual_one(recon_cgan, ax, m_size=marker_size)
    axs[3].append(ax)
    axsflat.append(ax)

    ax = fig.add_subplot(len(fig_names), len(indexs), nsample+16)
    laser_visual_one(recon_transformer, ax, m_size=marker_size)
    axs[4].append(ax)
    axsflat.append(ax)

    ax = fig.add_subplot(len(fig_names), len(indexs), nsample+20)
    laser_visual_one(l, ax, m_size=marker_size)
    axs[5].append(ax)
    axsflat.append(ax)

    nsample += 1


places = ['1', '2', '3', '4']
for i in range(len(places)):
    axs[0][i].set_title(places[i], fontsize=f_size)


for i, ax in enumerate(axsflat):
    ax.set_xlabel('meter', fontsize=f_size)
    ax.set_ylabel('%s' % (fig_names[i//len(places)]),
                  fontsize=f_size, rotation=0)
    ax.yaxis.set_label_coords(-0.5, 0.5)

for ax in axsflat:
    ax.label_outer()

# plt.show()
plt.savefig('visual_compare.png', dpi=200)
'''