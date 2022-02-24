from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import spectral_norm

from .ResnetEncoder import ResnetEncoder
from .MHA import MultiheadAttention
import matplotlib.pyplot as PLT

# Utils
class double_conv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch, filter_size=2):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(filter_size), double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class MergeMultimodal(nn.Module):
    """
    Merges features from multiple modalities (say RGB, projected occupancy) into a single
    feature map.
    """

    def __init__(self, nfeats, nmodes=2):
        super().__init__()
        self._nmodes = nmodes
        self.merge = nn.Sequential(
            nn.Conv2d(nmodes * nfeats, nfeats, 3, stride=1, padding=1),
            nn.BatchNorm2d(nfeats),
            nn.ReLU(),
            nn.Conv2d(nfeats, nfeats, 3, stride=1, padding=1),
            nn.BatchNorm2d(nfeats),
            nn.ReLU(),
            nn.Conv2d(nfeats, nfeats, 3, stride=1, padding=1),
        )

    def forward(self, *inputs):
        """
        Inputs:
            xi - (bs, nfeats, H, W)
        """
        x = torch.cat(inputs, dim=1)
        return self.merge(x)

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


class Encoder(nn.Module):
    """ Encodes the Image into low-dimensional feature representation

    Attributes
    ----------
    num_layers : int
        Number of layers to use in the ResNet
    img_ht : int
        Height of the input RGB image
    img_wt : int
        Width of the input RGB image
    pretrained : bool
        Whether to initialize ResNet with pretrained ImageNet parameters

    Methods
    -------
    forward(x, is_training):
        Processes input image tensors into output feature tensors
    """

    def __init__(self, num_layers, img_ht, img_wt, pretrained=True, all_features=False, in_channels=3):
        super(Encoder, self).__init__()

        self.resnet_encoder = ResnetEncoder(num_layers, pretrained)
        if in_channels != 3:
            weight1 = self.resnet_encoder.encoder.conv1.weight.clone()
            new_first_layer = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).requires_grad_()
            new_first_layer.weight[:,:min(3, in_channels),:,:].data[...] =  Variable(weight1[:,:min(3, in_channels),:,:], requires_grad=True)
            self.resnet_encoder.encoder.conv1 = new_first_layer

        num_ch_enc = self.resnet_encoder.num_ch_enc
        # convolution to reduce depth and size of features before fc
        self.conv1 = Conv3x3(num_ch_enc[-1], 128)
        self.conv2 = Conv3x3(128, 128)
        self.pool = nn.MaxPool2d(2)
        self.all_features = all_features

    def forward(self, x):
        """

        Parameters
        ----------
        x : torch.FloatTensor
            Batch of Image tensors
            | Shape: (batch_size, 3, img_height, img_width)

        Returns
        -------
        x : torch.FloatTensor
            Batch of low-dimensional image representations
            | Shape: (batch_size, 128, img_height/128, img_width/128)
        """
        features = []
        batch_size, c, h, w = x.shape
        features.extend(self.resnet_encoder(x))
        features.append(self.pool(self.conv1(features[-1])))
        features.append(self.conv2(features[-1]))
        # features.append(self.pool(features[-1]))

        if self.all_features:
            return features

        return features[-1]

class ChandrakarEncoder(nn.Module):
    def __init__(self, n_channels, scales, nsf=16):
        super().__init__()
        self.inc = double_conv(n_channels, nsf)
        self.down1 = down(nsf, nsf * 2, scales[0])
        self.down2 = down(nsf * 2, nsf * 4, scales[1])
        self.down3 = down(nsf * 4, nsf * 8, scales[2])
        self.down4 = down(nsf * 8, nsf * 8, scales[3])

    def forward(self, x):
        x1 = self.inc(x)  # (bs, nsf, ..., ...)
        x2 = self.down1(x1)  # (bs, nsf*2, ... ,...)
        x3 = self.down2(x2)  # (bs, nsf*4, ..., ...)
        x4 = self.down3(x3)  # (bs, nsf*8, ..., ...)
        x5 = self.down4(x4)  # (bs, nsf*8, ..., ...)

        return x5


class Decoder(nn.Module):
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

    def __init__(self, num_ch_enc, num_class=2, occ_map_size=256, type='', in_features=128):
        super(Decoder, self).__init__()
        self.num_output_channels = num_class
        self.occ_map_size = occ_map_size
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        # self.num_ch_dec = np.array([64, 128, 256])
        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            if type == 'transform_decoder':
                num_ch_in = in_features if i == 4 else self.num_ch_dec[i + 1]
            else:
                num_ch_in = in_features if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = nn.Conv2d(
                num_ch_in, num_ch_out, 3, 1, 1)
            self.convs[("norm", i, 0)] = nn.BatchNorm2d(num_ch_out)
            self.convs[("relu", i, 0)] = nn.ReLU(True)

            # upconv_1
            self.convs[("upconv", i, 1)] = nn.Conv2d(
                num_ch_out, num_ch_out, 3, 1, 1)
            self.convs[("norm", i, 1)] = nn.BatchNorm2d(num_ch_out)

        self.convs["topview"] = Conv3x3(
            self.num_ch_dec[0], self.num_output_channels)
        self.dropout = nn.Dropout3d(0.2)
        self.rescale_output = lambda x: F.interpolate(x, (self.occ_map_size, self.occ_map_size), mode='bilinear')
        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, x, is_training=True):
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
        h_x = 0
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = self.convs[("norm", i, 0)](x)
            x = self.convs[("relu", i, 0)](x)
            x = upsample(x)
            x = self.convs[("upconv", i, 1)](x)
            x = self.convs[("norm", i, 1)](x)

        x = self.convs["topview"](x)
        x = self.rescale_output(x)
        if not is_training:
            x = nn.Softmax2d()(x)

        return x

class Discriminator(nn.Module):
    """
    A patch discriminator used to regularize the decoder
    in order to produce layouts close to the true data distribution
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, 8, 3, 2, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(8, 16, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(16, 32, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(32, 8, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(8, 1, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """

        Parameters
        ----------
        x : torch.FloatTensor
            Batch of output Layouts
            | Shape: (batch_size, 2, occ_map_size, occ_map_size)

        Returns
        -------
        x : torch.FloatTensor
            Patch output of the Discriminator
            | Shape: (batch_size, 1, occ_map_size/16, occ_map_size/16)
        """

        return self.main(x)

class DiscriminatorAttention(nn.Module):
    """
    A patch discriminator used to regularize the decoder
    in order to produce layouts close to the true data distribution
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, 8, 3, 2, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(8, 16, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(16, 32, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.attention = MultiheadAttention(None, 32, 4, 32)

        self.main2 = nn.Sequential(
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(32, 8, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(8, 1, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """

        Parameters
        ----------
        x : torch.FloatTensor
            Batch of output Layouts
            | Shape: (batch_size, 2, occ_map_size, occ_map_size)

        Returns
        -------
        x : torch.FloatTensor
            Patch output of the Discriminator
            | Shape: (batch_size, 1, occ_map_size/16, occ_map_size/16)
        """

        feat = self.main(x)
        feat = self.attention(feat)
        out = self.main2(feat)

        return out

class ConditionalDiscriminator(nn.Module):
    """
    A patch discriminator used to regularize the decoder
    in order to produce layouts close to the true data distribution
    """
    def __init__(self):
        super(ConditionalDiscriminator, self).__init__()
        self.layer1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(4, 8, 3, 2, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer2 = nn.Sequential(
            # state size. (ndf) x 32 x 32
            nn.Conv2d(8, 16, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer3 = nn.Sequential(
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(16, 32, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer4 = nn.Sequential(
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(32, 8, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer5 = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(8, 1, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        """

        Parameters
        ----------
        x : torch.FloatTensor
            Batch of output Layouts
            | Shape: (batch_size, 2, occ_map_size, occ_map_size)

        y : torch.FloatTensor
            Batch of output Layouts
            | Shape: (batch_size, 3, occ_map_size, occ_map_size)

        Returns
        -------
        x : torch.FloatTensor
            Patch output of the Discriminator
            | Shape: (batch_size, 1, occ_map_size/16, occ_map_size/16)
        """

        img_input = torch.cat((x, y), 1) # Concatenate by channels

        out1 = self.layer1(img_input)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        result = self.layer5(out4)

        return result, [out1, out2, out3, out4]