import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as PLT
from torchvision import transforms


def to_cpu(tensor):
    return tensor.detach().cpu()


def _sigmoid(x):
    return torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)


def normalize_image(x, range=None):
    """Rescale image pixels to span range [0, 1]
    """
    if range is not None and isinstance(range, (list, tuple)) and len(range) == 2:
        mi, ma = range
    else:
        ma = float(x.max().cpu().data)
        mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    if torch.is_tensor(x):
        x = torch.clamp(x, mi, ma)
    else:
        x = np.clip(x, a_min=mi, a_max=ma)
    return (x - mi) / d

def invnormalize_imagenet(x):
    inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
            std=[1/0.229, 1/0.224, 1/0.255]
        )
    return inv_normalize(x)


def mean_precision(eval_segm, gt_segm, n_cl):
    check_size(eval_segm, gt_segm)
    # cl, n_cl = extract_classes(gt_segm)
    cl = np.arange(n_cl)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)
    mAP = [0] * n_cl
    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        n_ij = np.sum(curr_eval_mask)
        val = n_ii / float(n_ij)
        if math.isnan(val):
            mAP[i] = 0.
        else:
            mAP[i] = val
    # print(mAP)
    return mAP


def mean_IU(eval_segm, gt_segm, n_cl):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    # cl, n_cl = union_classes(eval_segm, gt_segm)
    # _, n_cl_gt = extract_classes(gt_segm)
    cl = np.arange(n_cl)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        # PLT.imshow(curr_eval_mask, cmap=PLT.cm.jet)
        # PLT.show()
        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)
        # print(n_ii, t_i, n_ij)
        # print('n_ii', n_ii)
        # print('t_i', t_i)
        # print('n_ij', n_ij)
        # print(n_ii, t_i, n_ij)
        IU[i] = n_ii / (t_i + n_ij - n_ii)

    # print('IU', IU)
    return IU

def compute_depth_errors(pred, gt, mask=None):
    """Computation of error metrics between predicted and ground truth depths
    """
    if mask is None:
        mask = np.ones_like(gt) == 1
        
    gt = gt[mask] + 1e-6
    pred = pred[mask] + 1e-6
        
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


'''
Auxiliary functions used during evaluation.
'''


def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]


def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask


def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl


def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _ = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)
    # n_cl = 9
    return cl, n_cl


def extract_masks(segm, cl, n_cl):
    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


'''
Exceptions
'''


class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
