import os

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from PIL import Image
import matplotlib.pyplot as PLT
import matplotlib.cm as mpl_color_map

from einops import rearrange


class MultiheadAttention(nn.Module):
    def __init__(self, patch_size, in_dim, heads, dim_head, dropout=0):
        super(MultiheadAttention, self).__init__()

        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == in_dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(in_dim, inner_dim, bias = False)
        self.to_k = nn.Linear(in_dim, inner_dim, bias = False)
        self.to_v = nn.Linear(in_dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, in_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        self.scores = None


    def forward(self, x_key, x_query=None, x_value=None):
        if x_query is None :
            x_query = x_key.clone()
        
        if x_value is None :
            x_value = x_key.clone()

        B, C, H, W = x_value.shape

        x_key = x_key.reshape((*x_key.shape[:2], -1)).transpose(-1,-2)  # Convert B x C x H X W -> B x HW x C
        x_query = x_query.reshape((*x_query.shape[:2], -1)).transpose(-1,-2)  # Convert B x C x H X W -> B x HW x C
        x_value = x_value.reshape((*x_value.shape[:2], -1)).transpose(-1,-2)  # Convert B x C x H X W -> B x HW x C

        k = self.to_k(x_key)
        q = self.to_q(x_query)
        v = self.to_v(x_value)

        k, q, v = map(lambda x : rearrange(x, 'b n (h d) -> b h n d', h=self.heads), [k, q, v])

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        
        self.scores = attn

        T = torch.matmul(attn, v)
        out = rearrange(T, 'b h n d -> b n (h d)') + x_value
        out = self.to_out(out)
        out = out.reshape((B, C, H, W))
        return out


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

    attention3 = MultiheadAttention(128)
    print(attention3(features, features2, features3).shape)
