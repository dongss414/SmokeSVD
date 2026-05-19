import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F




class TestScalarflowProcDataset(Dataset):
    def __init__(self, modelConfig, path, imagetransform=None,start_frame = 0, end_frame = 150):
        self.modelConfig = modelConfig
        self.size = modelConfig['den_size']
        self.velsize = modelConfig['vel_size']
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.ITEM = self.end_frame - self.start_frame + 1

        self.denpath = path
        self.imagetransform = imagetransform



    def resize3D(self, x, size):
        resized_data = F.interpolate(x, size=size, mode='trilinear',
                                     align_corners=False)
        return resized_data
    def bin2tensor(self, index):
        index = index + self.start_frame
        file = [os.path.join(self.denpath, f) for f in os.listdir(self.denpath) if f.endswith('.npz') and (f'imgs_{index:06d}' in f )]
        if not file:
            raise FileNotFoundError(f"No file found for imgs_{index:06d} in {self.denpath}")


        data = np.load(file[0])['data']
        data = torch.from_numpy(data)
        front = data[0:1,:,:,0] #[1920, 1080]
        side = data[3:4,:,:,0]
        # front = torch.flip(front, dims=[-1])
        # side = torch.flip(side, dims=[-1])
        front = torch.rot90(front, k=2, dims=(1, 2))
        side = torch.rot90(side, k=2, dims=(1, 2))
        top, left, bottom, right = 40, 115, 1040, 1895
        front = front[:, left:right, top:bottom]  # [1780, 1000]
        top, left, bottom, right = 40, 105, 1040, 1885
        side = side[:, left:right, top:bottom]  # [1780, 1000]
        side = torch.flip(side,dims=[-1])
        #test
        # front = data[2:3, :, :, 0]  # [1920, 1080]
        # front = torch.rot90(front, k=2, dims=(1, 2))
        # top, left, bottom, right = 00, 140, 1000, 1920
        # front = front[:, left:right, top:bottom]  # [1780, 1000]

        return front, side, index


    def __getitem__(self, index):

        front,side,num = self.bin2tensor(index)


        if self.imagetransform:
            front = self.imagetransform(front)
            side = self.imagetransform(side)


        den = torch.zeros(1,self.size[0], self.size[1], self.size[2])
        vel = torch.zeros(self.velsize)



        return {'front_view': front*2,
                'side_view': side*2,
                'den': den,
                'num': str(f'{num:06d}'),
                'vel': vel,
                }


    def __len__(self):
        return self.ITEM




def resize3D(x,size):

    resized_data = F.interpolate(x, size=size, mode='trilinear',
                                                align_corners=False)

    return resized_data

if __name__ == '__main__':
    pass

