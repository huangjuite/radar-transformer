import torch
import pandas as pd
import numpy as np
import pickle as pkl
import copy
from torch.utils.data.dataset import Dataset
from scipy.spatial.transform import Rotation as R



class RadarDataset(Dataset):
    def __init__(self):
        folder = '/media/ray/intelSSD/radar'
        date_time = '0805_1108'

        with open(folder+'/dataset_'+date_time+'.pkl', 'rb') as f:
            self.data = pkl.load(f)

        print('dataset loaded, length:%d'%len(self.data))

    def __getitem__(self, index):

        d = self.data[index]

        # t = d[0]
        lidar = torch.Tensor(d[1:242])
        radar = d[242:].reshape(-1, 7)

        radar = torch.Tensor(radar)

        return lidar, radar

    def __len__(self):
        return len(self.data)