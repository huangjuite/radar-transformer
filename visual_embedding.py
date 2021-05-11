import torch
import time
import numpy as np
import open3d as o3d
from utils import draw_dataset, draw_laser, filter, laser_to_pc, pc_to_laser
from torch.utils.tensorboard import SummaryWriter

from dataset import RadarDataset
from models import RadarEncoder
from network_v2 import Actor, ActorCNN

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


dataset = RadarDataset()

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu")


def get_data(indx=39714):
    l, rp0 = dataset[indx]
    _, rp1 = dataset[indx-1]
    _, rp2 = dataset[indx-2]
    _, rp3 = dataset[indx-3]
    _, rp4 = dataset[indx-4]

    rp = np.concatenate((rp0, rp1, rp2, rp3, rp4))
    rp = filter(rp)

    r_laser_np = pc_to_laser(rp[:, 3:6])

    return l, rp0, r_laser_np


indx = 39714
# rpcd = o3d.geometry.PointCloud()
# lpcd = o3d.geometry.PointCloud()
# frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
#     size=4, origin=[0, 0, 0])
# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(rpcd)
# vis.add_geometry(lpcd)
# vis.add_geometry(frame)


# for i in range(indx, indx+50):
#     l, r, rl = get_data(i)

#     rpc = r[:, 3:6]
#     lpc = laser_to_pc(l)

#     rpcd.points = o3d.utility.Vector3dVector(rpc)
#     lpcd.points = o3d.utility.Vector3dVector(lpc)

#     vis.update_geometry(rpcd)
#     vis.update_geometry(lpcd)
#     vis.poll_events()
#     vis.update_renderer()
#     # o3d.visualization.draw_geometries([lpcd, rpcd, frame])

#     time.sleep(0.5)

encoder_transformer = RadarEncoder().to(device)
encoder_transformer.load_state_dict(torch.load(
    'model/transformer/transformer_encoder_forward.pth'))
encoder_transformer.eval()

encoder_cnn = ActorCNN(
    device, 243, 2
).to(device)
encoder_cnn.load_state_dict(torch.load('model/rdpg_torch_forward/mse.pth'))

encoder = ActorCNN(
    device, 243, 2
).to(device)
encoder.load_state_dict(torch.load('model/s1536_f1869509.pth'))


writer = SummaryWriter('runs/vis_embedding')
embds = []
labels = []
imgs = []

for i in range(indx, indx+500):
    l, r, rl = get_data(i)

    r_t = torch.Tensor(r).to(device)
    r_t = torch.unsqueeze(r_t, dim=1)

    l = torch.Tensor(l).to(device)
    l = torch.unsqueeze(l, dim=0)

    rl = torch.Tensor(rl).to(device)
    rl = torch.unsqueeze(rl, dim=0)

    r_embd = encoder_transformer(r_t, None).detach().cpu()
    rl_embd = encoder(rl).detach().cpu()[0]
    l_embd = encoder(l).detach().cpu()[0]

    embds.append(r_embd)
    embds.append(rl_embd)
    embds.append(l_embd)
    labels.append('radar_point_%d' % (i-indx))
    labels.append('radar_scan_%d' % (i-indx))
    labels.append('lidar_scan_%d' % (i-indx))

    tune = (50+3*(i-indx))/255.
    zs = torch.ones((1, 1, 40, 40)) - tune
    cs = torch.ones((1, 1, 40, 40))
    label_r = torch.hstack((cs, zs, zs))
    label_g = torch.hstack((zs, cs, zs))
    label_b = torch.hstack((zs, zs, cs))
    imgs.append(label_r)
    imgs.append(label_g)
    imgs.append(label_b)

embds = torch.vstack(embds)
imgs = torch.cat(imgs)
print(imgs.shape)

writer.add_embedding(
    embds,
    metadata=labels,
    label_img=imgs
)
