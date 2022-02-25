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

import crossView

from utils import invnormalize_imagenet


class P_BasicTransformer(nn.Module):
    def __init__(self, models, opt):
        super(P_BasicTransformer, self).__init__()

        self.opt = opt
        self.pos_emb1D = torch.nn.Parameter(torch.randn(1, 128, 64), requires_grad=True)

        self.encoder = crossView.Encoder(18, self.opt.height, self.opt.width, True) # models["encoder"]
        self.basic_transformer = crossView.MultiheadAttention(None, 128, 4, 32) # models["BasicTransformer"]
        self.decoder = crossView.Decoder(
            self.encoder.resnet_encoder.num_ch_enc, self.opt.num_class, self.opt.occ_map_size, in_features=128) # models["decoder"]

        self.bottleneck = [self.basic_transformer.to_out]

    def forward(self, x):
        features = self.encoder(x)
        
        b, c, h, w = features.shape
        features = (features.reshape(b, c, -1) + self.pos_emb1D[:, :, :h*w]).reshape(b, c, h, w)

        features = self.basic_transformer(features, features, features)  # BasicTransformer

        topview = self.decoder(features)

        return topview