import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as PLT
import numpy as np
import cv2
import kornia.filters as ktf
from kornia.losses import dice_loss


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class compute_transform_losses(nn.Module):
    def __init__(self, device='GPU'):
        super(compute_transform_losses, self).__init__()
        self.device = device
        self.l1_loss = L1Loss()

    def forward(self, outputs, retransform_output):
        loss = F.l1_loss(outputs, retransform_output, size_average=False)
        return loss




class compute_losses(nn.Module):
    def __init__(self, device='GPU'):
        super(compute_losses, self).__init__()
        self.device = device
        self.L1Loss = nn.L1Loss()

    def forward(self, opt, weight, inputs, outputs, features=None, retransform_features=None):
        losses = {}
        type = opt.type
        # losses["topview_loss"] = 0
        # losses["transform_topview_loss"] = 0
        # losses["transform_loss"] = 0
        # losses["depth_loss"] = 0

        topview_key = 'static' if 'static' in inputs else 'static_gt'

        if "topview" in outputs:
            losses["topview_loss"] = self.compute_topview_loss(
                outputs["topview"],
                inputs[topview_key],
                weight[type])

        if "transform_topview" in outputs:
            losses["transform_topview_loss"] = self.compute_topview_loss(
                outputs["transform_topview"],
                inputs[topview_key],
                weight[type])

        if (features is not None) and (retransform_features is not None):
            losses["transform_loss"] = self.compute_transform_losses(
                features,
                retransform_features)

        if "pred_depth" in outputs:
            losses["depth_loss"] = self.compute_depth_losses(
                outputs["pred_depth"],
                inputs["depth_gt"])

        if "boundary" in inputs:
            losses["boundary"] = self.compute_boundary_loss(
                outputs["topview"],
                inputs["boundary"])

        # losses["loss"] = losses["topview_loss"] + 0.001 * losses["transform_loss"] + 1 * losses["transform_topview_loss"]
        losses["loss"] = losses["topview_loss"] #+ 0.1 * losses["boundary"]
        # losses["loss"] = losses["depth_loss"]

        return losses

    def compute_topview_loss(self, outputs, true_top_view, weight):
        generated_top_view = outputs
        true_top_view = torch.squeeze(true_top_view.long(), dim=1)
        loss = nn.CrossEntropyLoss(weight=torch.Tensor([1., 3, 2]).cuda())
        # loss = dice_loss
        output = loss(generated_top_view, true_top_view)
        return output.mean()

    def compute_transform_losses(self, outputs, retransform_output):
        loss = self.L1Loss(outputs, retransform_output)
        return loss

    def compute_depth_losses(self, outputs, true_depth):
        loss = 0
        min = torch.amin(true_depth, dim=[1,2], keepdim=True)
        max = torch.amax(true_depth, dim=[1,2], keepdim=True)
        rel_depth = (true_depth - min)/(max - min + 1e-6)
        for output in outputs:
            loss += self.L1Loss(torch.squeeze(output, dim=1), rel_depth)
        return loss

    def compute_boundary_loss(self, topview, boundary):
        gt = boundary.float()
        topview = nn.Softmax2d()(topview)  #Convert to prob
        occ_ch = topview[:, 1]
        explored_ch = 1 - topview[:, 0]  # 1 - unknown
        pred = torch.unsqueeze(occ_ch, dim=1)
        gradient_pred = torch.linalg.norm(2 * ktf.spatial_gradient(pred), dim=2) * torch.unsqueeze(explored_ch, dim=1)
        gradient_pred = torch.clamp(gradient_pred, min=0, max=1)
        return nn.L1Loss()(gradient_pred.squeeze(1), gt)