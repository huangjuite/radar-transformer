
import copy
import os
import torch
import pandas as pd
import numpy as np
import pickle as pkl
from tqdm import tqdm
import open3d as o3d

import torch.nn.utils.rnn as rnn_utils
from torch.utils.data.dataset import Dataset


# rnn_utils.pack_sequence
def collate_fn(batch):
    lidar = torch.Tensor([item[0] for item in batch])
    radar = [torch.Tensor(item[1]) for item in batch]

    lengths = torch.Tensor([len(r) for r in radar])

    radar = rnn_utils.pad_sequence(radar, batch_first=False, padding_value=0)
    radar = rnn_utils.pack_padded_sequence(
        radar, lengths, batch_first=False, enforce_sorted=False)

    return lidar, radar


class RadarDataset(Dataset):
    def __init__(self, folder='/media/ray/intelSSD/radar/pkl/', remove_oulier=None):

        paths = []
        allfiles = os.listdir(folder)
        files = []
        for f in allfiles:
            if f[-4:] == '.pkl':
                files.append(f)
        self.data = []
        t = tqdm(files)
        for f in :
            with open(folder+f, 'rb') as h:
                self.data.extend(pkl.load(h))
        t.set_description('dataset loaded, length:%d' % len(self.data))
        
        self.remove_oulier = remove_oulier
        if remove_oulier is not None:
            print('remove outlier=%f'%remove_oulier)

    def __getitem__(self, index):

        d = self.data[index]

        # t = d[0]
        lidar = d[1:242]
        radar = d[242:].reshape(-1, 7)

        if self.remove_oulier is not None:
            pt = radar[:, 3:6]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pt)
            cl, ind = pcd.remove_radius_outlier(nb_points=1, radius=self.remove_oulier)
            radar = radar[ind]

        return lidar, radar

    def __len__(self):
        return len(self.data)
