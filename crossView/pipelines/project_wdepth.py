###########################################################################
'''
PIPELINE FOR MERGING RGB AND CKDEPTH PRE-ATTENTION 

RGB -----> RESNET -> Fr----> CONCAT ---> MultiModal Merge --> Decoder
CKDEPTH -> RESNET -> Fck--|

'''
###########################################################################

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
from crossView.grad_cam import *

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from utils import invnormalize_imagenet

import crossView


class ProjectWDepth(nn.Module):
    def __init__(self, models, opt):
        super(ProjectWDepth, self).__init__()
        self.opt = opt

        # self.pos_emb1D = torch.nn.Parameter(torch.randn(1, 128, 256), requires_grad=True)

        self.encoder = crossView.Encoder(18, self.opt.height, self.opt.width, True) # models["encoder"]
        # self.basic_transformer = crossView.MultiheadAttention(None, 128, 4, 32) # models["BasicTransformer"]
        self.decoder = crossView.Decoder(
            self.encoder.resnet_encoder.num_ch_enc, self.opt.num_class, self.opt.occ_map_size, in_features=128) # models["decoder"]

        # self.bottleneck = [models["BasicTransformer"].to_out]

        self.setup_cam_coords()


    def setup_cam_coords(self):
        proj_xs, proj_ys = np.meshgrid(
            np.linspace(-1, 1, 16), np.linspace(1, -1, 16)
        )
        xs = proj_xs.reshape(-1)
        ys = proj_ys.reshape(-1)
        zs = -np.ones_like(xs)
        K = np.eye(3)  # since fov=90
        inv_K = np.linalg.inv(K)
        self.cam_coords = torch.nn.Parameter(torch.from_numpy(inv_K @ np.array([xs, ys, zs])).float(), requires_grad=False)


    def forward(self, inputs):
        
        rgb, depth = torch.split(inputs, 3, dim=1)

        features = self.encoder(rgb)

        b, c, h, w = features.shape
        # features = (features.reshape(b, c, -1) + self.pos_emb1D[:, :, :h*w]).reshape(b, c, h, w)

        depth = F.interpolate(input=depth, size=(w, h), mode='bilinear')
        pc = depth.reshape(b, 1, -1) * self.cam_coords.unsqueeze(dim=0).repeat(b, 1, 1)
        pc = pc.transpose(-1, -2)
        pc[..., 1] += self.opt.cam_height

        map_size = self.opt.occ_map_size // 4
        cell_size = 3.2/map_size
        max_height_idx = self.opt.obstacle_height // cell_size

        x_indices = (pc[..., 0]//cell_size).reshape(-1).long() + map_size//2
        y_indices = (pc[..., 1]//cell_size).reshape(-1).long()
        z_indices = (pc[..., 2]//cell_size).reshape(-1).long() + map_size
        batch_indices = torch.cat([torch.full([pc.shape[1]], ix, device=x_indices.device, dtype=torch.long) for ix in range(pc.shape[0])])

        valid_indices = (x_indices >= 0) & (x_indices < map_size) & (z_indices >= 0) & (z_indices < map_size) & (y_indices < max_height_idx)
        flat_idx = ((batch_indices * map_size * map_size * max_height_idx) + 
            (z_indices * map_size * max_height_idx) + 
            (x_indices * max_height_idx) + y_indices)[valid_indices]

        cell_agg_idx = ((batch_indices * map_size * map_size) + (z_indices * map_size) + x_indices)[valid_indices]

        rank = torch.argsort(flat_idx)
        flat_idx = flat_idx[rank]
        cell_agg_idx = cell_agg_idx[rank]

        kept = torch.ones_like(flat_idx, device=flat_idx.device, dtype=torch.bool)
        kept[:-1] = flat_idx[1:] != flat_idx[:-1]

        cell_kept = torch.ones_like(cell_agg_idx, device=flat_idx.device, dtype=torch.bool)
        cell_kept[:-1] = cell_agg_idx[1:] != cell_agg_idx[:-1]

        features = features.reshape((b, c, -1)).transpose(-2, -1).reshape((-1, c))
        features = features[valid_indices][rank]

        feature_sum = torch.cumsum(features, dim=0)
        x_sum = feature_sum[kept]
        x_sum = torch.cat([x_sum[:1], x_sum[1:] - x_sum[:-1]], dim=0)
        cell_agg_idx = cell_agg_idx[kept]

        keep_cell_highest = cell_kept[kept]
        x_sum = x_sum[keep_cell_highest]
        cell_kept_idx = cell_agg_idx[keep_cell_highest]

        warped_feature_grid = torch.zeros((b * map_size * map_size, c), dtype=torch.float32, device=x_sum.device, requires_grad=True).clone()
        warped_feature_grid[cell_kept_idx] = x_sum
        warped_feature_grid = warped_feature_grid.reshape((b, map_size, map_size, c)).permute((0, 3, 1, 2)) # B x C x Bh x Bw

        # features = self.basic_transformer(features, features, features)  # BasicTransformer

        topview = self.decoder(warped_feature_grid)

        return topview