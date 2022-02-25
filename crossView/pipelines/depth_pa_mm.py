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


class DepthPreAttnMerge(nn.Module):
    def __init__(self, models, opt):
        super(DepthPreAttnMerge, self).__init__()

        self.pos_emb1D = torch.nn.Parameter(torch.randn(1, 128, 64), requires_grad=True)

        self.encoder = models["encoder"]
        self.depth_encoder = models["DepthEncoder"]
        self.merge_multimodal = models["MergeMultimodal"]
        self.basic_transformer = models["BasicTransformer"]
        self.decoder = models["decoder"]

        self.opt = opt
        self.bottleneck = [models["BasicTransformer"].to_out]

    def forward(self, inputs):
        
        rgb, depth = torch.split(inputs, 3, dim=1)

        features = self.encoder(rgb)
        depth_features = self.depth_encoder(depth)

        features = self.merge_multimodal(features, depth_features)
        
        b, c, h, w = features.shape
        features = (features.reshape(b, c, -1) + self.pos_emb1D[:, :, :h*w]).reshape(b, c, h, w)

        features = self.basic_transformer(features, features, features)  # BasicTransformer

        topview = self.decoder(features)

        return topview