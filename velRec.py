import sys

import numpy as np
import torch
import torch.nn.functional as F
import math,time
from velModel import UNet3D_trainVelocity_loadDensity
from advection import *
from torchvision.utils import save_image
from render import *
from tqdm import tqdm

def render(den, light_positions=[(64, 64, -64)], intensity_p=0, intensity_a=1):

    w = 0.05

    front_view = render_with_y_axis_rotation_cuda(den * w, light_positions, intensity_p, intensity_a, n=0,
                                                  f=den.shape[0], rotation_angle=90)

    side_view = render_with_y_axis_rotation_cuda(den * w, light_positions, intensity_p, intensity_a, n=0,
                                                 f=den.shape[0], rotation_angle=180)
    front_view =  torch.rot90(front_view, k=2, dims=[0, 1])
    side_view =  torch.rot90(side_view, k=2, dims=[0, 1])

    return front_view, side_view

def saveImg(src_den,sr_den,frame,path):
    os.makedirs(path, exist_ok=True)
    src_front,src_side = render(src_den,light_positions=[src_den.shape[0], src_den.shape[1], src_den.shape[2]])
    sr_front,sr_side = render(sr_den,light_positions=[src_den.shape[0], src_den.shape[1], src_den.shape[2]])
    combined_image = torch.cat([src_front,sr_front,src_side,sr_side],dim=-1)
    # combined_image = torch.cat([src_front,src_side],dim=-1)
    # combined_image = torch.rot90(combined_image, k=2, dims=[0, 1])
    combined_image = F.interpolate(combined_image.unsqueeze(0).unsqueeze(0), scale_factor=4, mode='bilinear',
                                   align_corners=False).squeeze(0)
    save_image(combined_image, os.path.join(path, f"trainSample{frame:06d}.png"), nrow=1)




def srVel(density,nxt_density,velocity,source,model,isFirst,res=16,up=4):
    device = torch.device("cuda:0")
    if isFirst:
        sr_density = F.interpolate(density, scale_factor=up, mode='trilinear', align_corners=False)
    else:
        sr_density = density
    sr_source = F.interpolate(source.unsqueeze(0).unsqueeze(0), scale_factor=up, mode='trilinear', align_corners=False)
    nxt_density = F.interpolate(nxt_density, scale_factor=up, mode='trilinear', align_corners=False)[0, 0]

    C, D, H, W = velocity.shape

    x = math.ceil(D / (res-2))
    y = math.ceil(H / (res-2))
    z = math.ceil(W / (res-2))
    x_start_index = [i * (res-2) for i in range(x)]
    y_start_index = [i * (res-2) for i in range(y)]
    z_start_index = [i * (res-2) for i in range(z)]
    x_end_index = [i * (res-2) - 1 for i in range(1, x)]
    y_end_index = [i * (res-2) - 1 for i in range(1, y)]
    z_end_index = [i * (res-2) - 1 for i in range(1, z)]
    x_end_index.append(D - 1)
    y_end_index.append(H - 1)
    z_end_index.append(W - 1)

    sr_velocity = torch.zeros(
        (velocity.shape[0], velocity.shape[1] * up, velocity.shape[2] * up, velocity.shape[3] * up)).to(device)

    for i in range(x):
        for j in range(y):
            for k in range(z):
                d0 = x_start_index[i] - 1 if x_start_index[i] - 1 >= 0 else 0
                d1 = x_end_index[i] + 1 if x_end_index[i] + 1 < D else D - 1
                h0 = y_start_index[j] - 1 if y_start_index[j] - 1 >= 0 else 0
                h1 = y_end_index[j] + 1 if y_end_index[j] + 1 < H else H - 1
                w0 = z_start_index[k] - 1 if z_start_index[k] - 1 >= 0 else 0
                w1 = z_end_index[k] + 1 if z_end_index[k] + 1 < W else W - 1

                block = velocity[:, d0:d1 + 1, h0:h1 + 1, w0:w1 + 1]

                load_density = torch.cat((sr_density[:, :, d0:d1 + 1, h0:h1 + 1, w0:w1 + 1],
                                          nxt_density[d0:d1 + 1, h0:h1 + 1, w0:w1 + 1].unsqueeze(0).unsqueeze(0)),
                                         dim=1)[0]

                pad_d, pad_w, pad_h = 0, 0, 0
                if block.shape[1] != res:
                    pad_d = res - block.shape[1]
                    if d0 == 0:
                        block = F.pad(block, (0, 0, 0, 0, pad_d, 0), mode="constant", value=0)
                        load_density = F.pad(load_density, (0, 0, 0, 0, pad_d, 0), mode="constant", value=0)
                    elif d1 + 1 == D:
                        block = F.pad(block, (0, 0, 0, 0, 0, pad_d), mode="constant", value=0)
                        load_density = F.pad(load_density, (0, 0, 0, 0, 0, pad_d), mode="constant", value=0)
                if block.shape[2] != res:
                    pad_h = res - block.shape[2]
                    if h0 == 0:
                        block = F.pad(block, (0, 0, pad_h, 0, 0, 0), mode="constant", value=0)
                        load_density = F.pad(load_density, (0, 0, pad_h, 0, 0, 0), mode="constant", value=0)
                    elif h1 + 1 == H:
                        block = F.pad(block, (0, 0, 0, pad_h, 0, 0), mode="constant", value=0)
                        load_density = F.pad(load_density, (0, 0, 0, pad_h, 0, 0), mode="constant", value=0)
                if block.shape[3] != res:
                    pad_w = res - block.shape[3]
                    if w0 == 0:
                        block = F.pad(block, (pad_w, 0, 0, 0, 0, 0), mode="constant", value=0)
                        load_density = F.pad(load_density, (pad_w, 0, 0, 0, 0, 0), mode="constant", value=0)
                    elif w1 + 1 == W:
                        block = F.pad(block, (0, pad_w, 0, 0, 0, 0), mode="constant", value=0)
                        load_density = F.pad(load_density, (0, pad_w, 0, 0, 0, 0), mode="constant", value=0)
                # print(pad_d ,pad_h, pad_w)

                # super resolution
                with torch.no_grad():

                    # sr_block,_,_,_,_ = model(block.unsqueeze(0))

                    sr_block, _, _, _, _ = model(block.unsqueeze(0), load_density.unsqueeze(0))

                    # sr_block = F.interpolate(block.unsqueeze(0),scale_factor=4,mode='trilinear',align_corners=False)
                    if pad_d != 0 and d0 != 0:
                        d0_sr = up
                        d1_sr = d0_sr + (res-1 - pad_d) * up - 1

                    else:
                        d0_sr = up
                        d1_sr = 64-up-1

                    if pad_h != 0 and h0 != 0:
                        h0_sr = up
                        h1_sr = h0_sr + (res-1 - pad_h) * up - 1
                    else:
                        h0_sr = up
                        h1_sr = 64-up-1

                    if pad_w != 0 and w0 != 0:
                        w0_sr = up
                        w1_sr = w0_sr + (res-1 - pad_w) * up - 1
                    else:
                        w0_sr = up
                        w1_sr = 64-up-1
                    # print()

                    sr_block = sr_block[0, :, d0_sr:d1_sr + 1, h0_sr:h1_sr + 1, w0_sr:w1_sr + 1]
                    d0 = x_start_index[i] * up
                    d1 = d0 + sr_block.shape[1]
                    h0 = y_start_index[j] * up
                    h1 = h0 + sr_block.shape[2]
                    w0 = z_start_index[k] * up
                    w1 = w0 + sr_block.shape[3]

                    sr_velocity[:, d0:d1, h0:h1, w0:w1] = sr_block

    sr_nxt_density = advection(sr_density[0, 0], sr_velocity, sr_source[0, 0])
    sr_density = sr_nxt_density.unsqueeze(0).unsqueeze(0)

    return sr_density


