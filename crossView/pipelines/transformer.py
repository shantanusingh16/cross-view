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

from crossView.model import FeedForward

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

        self.scores = None

        self.bottleneck = [self.basic_transformer]

    def get_attention_map(self):
        return self.scores.mean(dim=1)

    def forward(self, x):
        features = self.encoder(x)
        
        b, c, h, w = features.shape
        features = (features.reshape(b, c, -1) + self.pos_emb1D[:, :, :h*w]).reshape(b, c, h, w)

        features = self.basic_transformer(features, features, features)  # BasicTransformer

        topview = self.decoder(features)
        self.scores = self.basic_transformer.scores

        return topview


#################################### Transformer Multiblock ########################################

class MultiBlockTransformer(nn.Module):
    def __init__(self, models, opt, nblocks=1):
        super(MultiBlockTransformer, self).__init__()

        self.opt = opt
        self.pos_emb1D = torch.nn.Parameter(torch.randn(1, 128, 64), requires_grad=True)

        self.encoder = crossView.Encoder(18, self.opt.height, self.opt.width, True) # models["encoder"]
        blocks = []
        for _ in range(nblocks):
            blocks.append(crossView.MultiheadAttention(None, 128, 4, 32, dropout=0.3)),
            blocks.append(crossView.FeedForward(64, 64, skip_conn=True, dropout=0.3)
        )
        self.transformer = nn.Sequential(*blocks)
        self.decoder = crossView.Decoder(
            self.encoder.resnet_encoder.num_ch_enc, self.opt.num_class, self.opt.occ_map_size, in_features=128) # models["decoder"]

        self.bottleneck = [blocks[-1]]
        self.scores = []
        
    def get_attention_map(self):
        return self.scores
    
    def forward(self, x):
        features = self.encoder(x)
        
        b, c, h, w = features.shape
        features = (features.reshape(b, c, -1) + self.pos_emb1D[:, :, :h*w]).reshape(b, c, h, w)

        features = self.transformer(features) 
        
        topview = self.decoder(features)

        self.scores = self.transformer._modules['0'].scores

        return topview
    
    
class BasicTransformer_Old(nn.Module):
    def __init__(self, models, opt):
        super(BasicTransformer_Old, self).__init__()

        self.opt = opt

        self.encoder = crossView.Encoder(18, self.opt.height, self.opt.width, True) # models["encoder"]
        self.BasicTransformer = crossView.BasicTransformer(8, 128) # models["BasicTransformer"]
        self.decoder = crossView.Decoder(
            self.encoder.resnet_encoder.num_ch_enc, self.opt.num_class, self.opt.occ_map_size, in_features=128) # models["decoder"]

        self.bottleneck = [self.BasicTransformer.merge2]
        self.scores = None

    def get_attention_map(self):
        return self.scores


    def forward(self, x):
        features = self.encoder(x)
        
        b, c, h, w = features.shape

        features = self.BasicTransformer(features)  # BasicTransformer

        self.scores = self.BasicTransformer.scores

        topview = self.decoder(features)

        return topview