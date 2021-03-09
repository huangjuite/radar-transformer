import torch
import pandas as pd
import numpy as np
import pickle as pkl
import copy

import torch.nn.utils.rnn as rnn_utils
from torch.utils.data.dataset import Dataset
from scipy.spatial.transform import Rotation as R


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
    def __init__(self):
        folder = '/media/ray/intelSSD/radar'
        date_time = '0805_1108'

        with open(folder+'/dataset_'+date_time+'.pkl', 'rb') as f:
            self.data = pkl.load(f)

        print('dataset loaded, length:%d' % len(self.data))

    def __getitem__(self, index):

        d = self.data[index]

        # t = d[0]
        lidar = d[1:242]
        radar = d[242:].reshape(-1, 7)

        return lidar, radar

    def __len__(self):
        return len(self.data)
