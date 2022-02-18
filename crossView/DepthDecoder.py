from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from crossView.model import Conv3x3, up, upsample, double_conv


class DepthDecoder(nn.Module):
    """ Encodes the Image into low-dimensional feature representation

    Attributes
    ----------
    num_ch_enc : list
        channels used by the ResNet Encoder at different layers

    Methods
    -------
    forward(x, ):
        Processes input image features into output occupancy maps/layouts
    """

    def __init__(self, num_ch_enc, height, width, type=''):
        super(DepthDecoder, self).__init__()
        self.height = height
        self.width = width
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        # self.num_ch_dec = np.array([64, 128, 256])
        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = 128 if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = nn.Conv2d(
                num_ch_in, num_ch_out, 3, 1, 1)
            self.convs[("norm", i, 0)] = nn.BatchNorm2d(num_ch_out)
            self.convs[("relu", i, 0)] = nn.ReLU(True)

            # upconv_1
            self.convs[("upconv", i, 1)] = up(num_ch_out + self.num_ch_enc[i], num_ch_out)

            self.convs[("depth", i, 1)] = nn.Sequential(double_conv(num_ch_out, num_ch_out), Conv3x3(num_ch_out, 1))
            
        self.dropout = nn.Dropout3d(0.2)
        self.rescale_output = lambda x: F.interpolate(x, (self.height, self.width), mode='bilinear')
        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, x, enc_features, is_training=True):
        """

        Parameters
        ----------
        x : torch.FloatTensor
            Batch of encoded feature tensors
            | Shape: (batch_size, 128, occ_map_size/2^5, occ_map_size/2^5)
        is_training : bool
            whether its training or testing phase

        Returns
        -------
        x : torch.FloatTensor
            Batch of output Layouts
            | Shape: (batch_size, 2, occ_map_size, occ_map_size)
        """
        outputs = []
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = self.convs[("norm", i, 0)](x)
            x = self.convs[("relu", i, 0)](x)
            x = self.convs[("upconv", i, 1)](x, enc_features[i])
            
            y = self.convs[("depth", i, 1)](x)
            y = self.rescale_output(y)

            if not is_training:
                y = nn.Sigmoid()(y)

            outputs.append(y)

        return outputs
