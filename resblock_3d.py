import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Res_Bottle_Block_down(nn.Module):
    def __init__(self, input, output, kernel_size, stride, padding, bias=True):
        super(Res_Bottle_Block_down, self).__init__()
        middle = output//4
        self.conv1 = nn.Conv3d(input, middle, kernel_size=1, stride=1, padding=0, bias=bias)
        self.conv2 = nn.Conv3d(middle, middle, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv3 = nn.Conv3d(middle, output, kernel_size=1, stride=1, padding=0, bias=bias)
        self.downsample = nn.Conv3d(input, output, kernel_size=stride, stride=stride, padding=0, bias=bias)
    def forward(self, x):
        xx = self.downsample(x)
        x = self.conv1(F.leaky_relu(x,negative_slope=0.2))
        x = self.conv2(F.leaky_relu(x,negative_slope=0.2))
        x = self.conv3(F.leaky_relu(x,negative_slope=0.2))
        x = x + xx
        return x

class Res_Block_down(nn.Module):
    def __init__(self, input, output, kernel_size, stride, padding, bias=True):
        super(Res_Block_down, self).__init__()
        self.conv1 = nn.Conv3d(input, output, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv2 = nn.Conv3d(output, output, kernel_size=kernel_size, stride=1, padding=padding, bias=bias)
        self.downsample = nn.Conv3d(input, output, kernel_size=stride, stride=stride, padding=0, bias=bias)
    def forward(self, x):
        xx = self.downsample(x)
        x = self.conv1(F.leaky_relu(x,negative_slope=0.2))
        x = self.conv2(F.leaky_relu(x,negative_slope=0.2))
        x = x + xx
        return x

class Res_Bottle_Block_up(nn.Module):
    def __init__(self, input, output, kernel_size, stride, padding, output_padding=1, bias=True):
        super(Res_Bottle_Block_up, self).__init__()
        middle = output//4
        self.conv1 = nn.Conv3d(input, middle, kernel_size=1, stride=1, padding=0, bias=bias)
        self.conv2 = nn.ConvTranspose3d(middle, middle, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        self.conv3 = nn.Conv3d(middle, output, kernel_size=1, stride=1, padding=0, bias=bias)
        self.upsample = nn.ConvTranspose3d(input, output, kernel_size=stride, stride=stride, padding=0, output_padding=0, bias=bias)

    def forward(self, x):
        xx = self.upsample(x)
        x = self.conv1(F.leaky_relu(x,negative_slope=0.2))
        x = self.conv2(F.leaky_relu(x,negative_slope=0.2))
        x = self.conv3(F.leaky_relu(x,negative_slope=0.2))
        x = x + xx
        return x

class Res_Block_up(nn.Module):
    def __init__(self, input, output, kernel_size, stride, padding, output_padding=1, bias=True):
        super(Res_Block_up, self).__init__()
        self.conv1 = nn.ConvTranspose3d(input, output, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        self.conv2 = nn.Conv3d(output, output, kernel_size=kernel_size, stride=1, padding=padding, bias=bias)
        self.upsample = nn.ConvTranspose3d(input, output, kernel_size=stride, stride=stride, padding=0, output_padding=0, bias=bias)

    def forward(self, x):
        xx = self.upsample(x)
        x = self.conv1(F.leaky_relu(x,negative_slope=0.2))
        x = self.conv2(F.leaky_relu(x,negative_slope=0.2))
        x = x + xx
        return x

class Res_Bottle_Block(nn.Module):
    def __init__(self, input, output, kernel_size, stride, padding, bias=True,dropout_prob=0):
        super(Res_Bottle_Block, self).__init__()
        middle = output//4
        self.conv1 = nn.Conv3d(input, middle, kernel_size=1, stride=1, padding=0, bias=bias)
        self.conv2 = nn.Conv3d(middle, middle, kernel_size=kernel_size, stride=1, padding=padding, bias=bias)
        self.conv3 = nn.Conv3d(middle, output, kernel_size=1, stride=1, padding=0, bias=bias)
        self.sample = nn.Conv3d(input, output, kernel_size=1, stride=1, padding=0, bias=bias)
        self.dropout = nn.Dropout3d(p=dropout_prob)
        
    def forward(self, x):
        xx = self.sample(x)
        x = self.conv1(F.leaky_relu(x,negative_slope=0.2))
        x = self.dropout(x)
        x = self.conv2(F.leaky_relu(x,negative_slope=0.2))
        x = self.dropout(x)
        x = self.conv3(F.leaky_relu(x,negative_slope=0.2))
        x = x + xx
        return x

class Res_Block(nn.Module):
    def __init__(self, input, output, kernel_size, stride, padding, bias=True):
        super(Res_Block, self).__init__()
        self.conv1 = nn.Conv3d(input, output, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv2 = nn.Conv3d(output, output, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.sample = nn.Conv3d(input, output, kernel_size=1, stride=1, padding=0, bias=bias)
        
    def forward(self, x):
        xx = self.sample(x)
        x = self.conv1(F.leaky_relu(x,negative_slope=0.2))
        x = self.conv2(F.leaky_relu(x,negative_slope=0.2))
        x = x + xx
        return x