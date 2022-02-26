import os

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from PIL import Image
import matplotlib.pyplot as PLT
import matplotlib.cm as mpl_color_map

from crossView.CycledViewProjection import TransformModule


class  BasicTransformer(nn.Module):
    def __init__(self, patch_size, in_dim):
        super(BasicTransformer, self).__init__()
        
        self.mpl_head1 = nn.Sequential(nn.BatchNorm2d(in_dim), TransformModule(dim=patch_size))
        self.mpl_head2 = nn.Sequential(nn.BatchNorm2d(in_dim), TransformModule(dim=patch_size))
        
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        
        # self.fc = nn.Sequential(nn.Linear(patch_size, patch_size), nn.Dropout(p=0.5))

        self.merge1 = nn.Conv2d(in_channels=in_dim * 2, out_channels=in_dim, kernel_size=1)
        self.merge2 = nn.Conv2d(in_channels=in_dim * 2, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        
        self.scores = None

    def forward(self, front_x):
        features = self.mpl_head1(front_x)
        
        m_batchsize, C, width, height = features.size()
        proj_query = self.query_conv(features).view(m_batchsize, -1, width * height)  # B x C x (N)
        proj_key = self.key_conv(features).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B x C x (W*H)

        # proj_query = proj_query/torch.linalg.norm(proj_query, ord=2, dim=-1, keepdim=True)
        # proj_key = proj_key/torch.linalg.norm(proj_key, ord=2, dim=-1, keepdim=True)

        energy = torch.bmm(proj_key, proj_query) / (C ** 0.5) # transpose check
        energy = self.softmax(energy)
        proj_value = self.value_conv(features).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B x C x N

        V = torch.bmm(energy, proj_value).permute(0, 2, 1).view(m_batchsize, -1, width, height)

        T = self.merge1(torch.cat((features, V), dim=1))  # Skip connection 1
        
        front_res = self.mpl_head2(T)

        output = self.merge2(torch.cat((front_res, V), dim=1)) # Skip connection 2
        
        self.scores = energy
        
        return output


if __name__ == '__main__':
    # features = torch.arange(0, 24)
    features = torch.arange(0, 65536)
    features = torch.where(features < 20, features, torch.zeros_like(features))
    # features = features.view([2, 3, 4]).float()
    features = features.view([8, 128, 8, 8]).float()

    features2 = torch.arange(0, 65536)
    features2 = torch.where(features2 < 20, features2, torch.zeros_like(features2))
    # features = features.view([2, 3, 4]).float()
    features2 = features2.view([8, 128, 8, 8]).float()

    features3 = torch.arange(0, 65536)
    features3 = torch.where(features3 < 20, features3, torch.zeros_like(features3))
    # features = features.view([2, 3, 4]).float()
    features3 = features3.view([8, 128, 8, 8]).float()

    attention3 = BasicTransformer(128)
    print(attention3(features, features2, features3).shape)
