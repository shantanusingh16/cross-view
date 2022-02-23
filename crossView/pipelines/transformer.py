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


class P_BasicTransformer(nn.Module):
    def __init__(self, models, opt):
        super(P_BasicTransformer, self).__init__()

        self.encoder = models["encoder"]
        self.basic_transformer = models["BasicTransformer"]
        self.decoder = models["decoder"]
        self.transform_decoder = models["transform_decoder"]

        self.opt = opt
        self.bottleneck = [models["BasicTransformer"].merge2]

    def forward(self, x):
        features = self.encoder(x)

        x_feature = retransform_features = transform_feature = features #= depth_features
        features = self.basic_transformer(features)  # BasicTransformer

        topview = self.decoder(features)

        return topview