
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
    def __init__(
        self,
        folder='./data/pkl/',
        remove_oulier=None
    ):

        paths = []
        allfiles = os.listdir(folder)
        files = []
        for f in allfiles:
            if f[-4:] == '.pkl':
                files.append(f)
        self.data = []
        t = tqdm(files)
        for f in t:
            with open(folder+f, 'rb') as h:
                self.data.extend(pkl.load(h))
        print('\ndataset loaded, length:%d' % len(self.data))

        self.remove_oulier = remove_oulier
        if remove_oulier is not None:
            print('remove outlier=%f' % remove_oulier)

    def __getitem__(self, index):

        d = self.data[index]

        # t = d[0]
        lidar = d[1:242]
        radar = d[242:].reshape(-1, 7)

        if self.remove_oulier is not None:
            pt = radar[:, 2:5]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pt)
            cl, ind = pcd.remove_radius_outlier(
                nb_points=1, radius=self.remove_oulier)
            radar = radar[ind]

        return lidar, radar

    def __len__(self):
        return len(self.data)


class RadarDatasetClass(Dataset):
    def __init__(
        self,
        folder='./data/pkl/',
        remove_oulier=None,
        scene='corridor',
    ):
        scene_files = {
            'corridor': [
                '0717_1411',
                '0720_1129',
                '0722_1428',
                '0722_1437',
                '0722_1448',
                '0722_1500',
                '0722_1506',
                '0722_1510',
                '0722_1514',
                '0724_1101',
                '0724_1132',
                '0724_1204',
                '0727_1005',
                '0727_1027',
                '0727_1102',
                '0727_1435',
                '0727_1452',
                '0727_1510',
                '0727_1520',
                '0805_1108',
                '0805_1127',
                '0805_1147'
            ],
            'parking': [
                '0717_1504',
                '0720_1105',
                '0805_1349',
                '0805_1425'
            ]
        }
        paths = []
        allfiles = os.listdir(folder)
        allfiles.sort()
        files = []
        print('scene: ', scene)
        for f in allfiles:
            if f[-4:] == '.pkl' and any(s in f for s in scene_files[scene]):
                files.append(f)
        self.data = []
        t = tqdm(files)
        for f in t:
            with open(folder+f, 'rb') as h:
                self.data.extend(pkl.load(h))
        print('\ndataset loaded, length:%d' % len(self.data))

        self.remove_oulier = remove_oulier
        if remove_oulier is not None:
            print('remove outlier=%f' % remove_oulier)

    def __getitem__(self, index):

        d = self.data[index]

        # t = d[0]
        lidar = d[1:242]
        radar = d[242:].reshape(-1, 7)

        if self.remove_oulier is not None:
            pt = radar[:, 2:5]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pt)
            cl, ind = pcd.remove_radius_outlier(
                nb_points=1, radius=self.remove_oulier)
            radar = radar[ind]

        return lidar, radar

    def __len__(self):
        return len(self.data)
