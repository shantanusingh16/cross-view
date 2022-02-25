import argparse
import os
import random
import time
from pathlib import Path

import crossView

import cv2
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import autograd
# from torch.optim.lr_scheduler import ExponentialLR
# from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import MultiStepLR
# from torch.optim.lr_scheduler import CosineAnnealingLR
from tensorboardX import SummaryWriter

import torch.nn.functional as F

from PIL import Image
import matplotlib.pyplot as PLT
import matplotlib.cm as mpl_color_map

from opt import get_args

import tqdm

from losses import compute_losses
from utils import mean_IU, mean_precision, normalize_image, invnormalize_imagenet

import pickle

from einops import rearrange

from crossView.pipelines.transformer import P_BasicTransformer
from crossView.pipelines.depth_pa_mm import DepthPreAttnMerge
from crossView.pipelines.project_wdepth import ProjectWDepth


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    if os.path.islink(filename):
        filename = os.readlink(filename)
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


class Trainer:
    def __init__(self):
        self.opt = get_args()
        self.models = {}
        self.weight = {"static": self.opt.static_weight, "dynamic": self.opt.dynamic_weight}
        self.seed = self.opt.global_seed
        self.device = "cuda"
        self.criterion_d = nn.BCEWithLogitsLoss()
        self.parameters_to_train = []
        self.transform_parameters_to_train = []
        self.detection_parameters_to_train = []
        self.base_parameters_to_train = []
        self.parameters_to_train = []
        self.parameters_to_train_D = []
        self.criterion = compute_losses()
        self.create_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        self.epoch = 0
        self.start_epoch = 0
        self.scheduler = 0
        # Save log and models path
        # self.opt.log_root = os.path.join(self.opt.log_root, self.opt.split)
        # self.opt.save_path = os.path.join(self.opt.save_path, self.opt.split)
        if self.opt.split == "argo":
            self.opt.log_root = os.path.join(self.opt.log_root, self.opt.type)
            self.opt.save_path = os.path.join(self.opt.save_path, self.opt.type)
        self.writer = SummaryWriter(os.path.join(self.opt.log_root, self.opt.model_name, self.create_time))
        self.log = open(os.path.join(self.opt.log_root, self.opt.model_name, self.create_time,
                                     '%s.csv' % self.opt.model_name), 'w')

        if self.seed != 0:
            self.set_seed()  # set seed

        # Initializing models
        # self.pos_emb1D = torch.nn.Parameter(torch.randn(1, 128, 64), requires_grad=True)
        # self.base_parameters_to_train.append(self.pos_emb1D)

        self.models["encoder"] = crossView.Encoder(18, self.opt.height, self.opt.width, True)
        # self.models["BasicTransformer"] = crossView.BasicTransformer(8, 128)
        # self.models["BasicTransformer2"] = crossView.BasicTransformer2(8, 128)
        # self.models["BasicTransformer"] = crossView.MultiheadAttention(None, 128, 4, 32)
        
        
        if self.opt.chandrakar_input_dir != "None":
            self.multimodal_input = True
            # self.models["ChandrakarEncoder"] = crossView.ChandrakarEncoder(2, [4,4,2,2], 16)

            # self.models["ChandrakarEncoder"] = crossView.ChandrakarEncoder(1, [2,2,2,2], 16)
            # self.models['CycledViewProjectionMultimodal'] = crossView.CycledViewProjectionMultimodal(in_dim=8, in_channels=128)

            self.models["ChandrakarEncoder"] = crossView.Encoder(18, self.opt.height, self.opt.width, False, False, 2)

            # To project the chandrakar features from 1 ch to 128 ch (same as resnet dim)
            # lambda x: rearrange(x, "b c (x p_x) (y p_y) -> b (p_x p_y c) x y", p_x=patch_size, p_y=patch_size)
            # self.models["ChandrakarEncoder"] = crossView.ChandrakarEncoder(1, [2,2,1,1], 16)
            # self.models["ChandrakarAttn"] = crossView.MultiheadAttention(None, 128, 4, 32)
        else:
            self.multimodal_input = False

        # self.models["MergeMultimodal"] = crossView.MergeMultimodal(128, 2)
        # self.models["DepthEncoder"] = crossView.Encoder(18, self.opt.height, self.opt.width, False, False, 1)

        # self.models['CycledViewProjection'] = crossView.CycledViewProjection(in_dim=8)
        # self.models["CrossViewTransformer"] = crossView.CrossViewTransformer(128)

        self.models["decoder"] = crossView.Decoder(
            self.models["encoder"].resnet_encoder.num_ch_enc, self.opt.num_class, self.opt.occ_map_size, in_features=128)
        # self.models["transform_decoder"] = crossView.Decoder(
        #     self.models["encoder"].resnet_encoder.num_ch_enc, self.opt.num_class, self.opt.occ_map_size, "transform_decoder")

        self.pipeline = ProjectWDepth(self.models, self.opt)
        self.pipeline.to(self.device)
        self.base_parameters_to_train += list(self.pipeline.parameters())

        # for key in self.models.keys():
        #     self.models[key].to(self.device)
        #     if "discr" in key:
        #         self.parameters_to_train_D += list(
        #             self.models[key].parameters())
        #     elif "transform" in key:
        #         self.transform_parameters_to_train += list(self.models[key].parameters())
        #     else:
        #         self.base_parameters_to_train += list(self.models[key].parameters())
        self.parameters_to_train = [
            {"params": self.transform_parameters_to_train, "lr": self.opt.lr_transform},
            {"params": self.base_parameters_to_train, "lr": self.opt.lr},
        ]

        if self.opt.grad_clip_value is not None:
            for p in self.transform_parameters_to_train:
                p.register_hook(lambda grad: torch.clamp(grad, -self.opt.grad_clip_value, self.opt.grad_clip_value))
            for p in self.base_parameters_to_train:
                p.register_hook(lambda grad: torch.clamp(grad, -self.opt.grad_clip_value, self.opt.grad_clip_value))

        # Optimization
        self.model_optimizer = optim.Adam(
            self.parameters_to_train)
        # self.scheduler = ExponentialLR(self.model_optimizer, gamma=0.98)
        # self.scheduler = StepLR(self.model_optimizer, step_size=step_size, gamma=0.65)
        self.scheduler = MultiStepLR(self.model_optimizer, milestones=self.opt.lr_steps, gamma=0.1)
        # self.scheduler = CosineAnnealingLR(self.model_optimizer, T_max=15)  # iou 35.55

        self.patch = (1, self.opt.occ_map_size // 2**4, self.opt.occ_map_size // 2**4)

        # Data Loaders
        dataset_dict = {
            "3Dobject": crossView.KITTIObject,
            "odometry": crossView.KITTIOdometry,
            "argo": crossView.Argoverse,
            "raw": crossView.KITTIRAW,
            "gibson": crossView.GibsonDataset,
            "gibson4": crossView.Gibson4Dataset
        }

        self.dataset = dataset_dict[self.opt.split]
        fpath = os.path.join(
            os.path.dirname(__file__),
            "splits",
            self.opt.split,
            "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        self.val_filenames = val_filenames
        self.train_filenames = train_filenames

        train_dataset = self.dataset(self.opt, train_filenames)
        val_dataset = self.dataset(self.opt, val_filenames, is_train=False)

        self.train_loader = DataLoader(
            train_dataset,
            self.opt.batch_size,
            True,
            num_workers=self.opt.train_workers,
            pin_memory=True,
            drop_last=True)
        self.val_loader = DataLoader(
            val_dataset,
            self.opt.batch_size,
            True,
            num_workers=self.opt.val_workers,
            pin_memory=True,
            drop_last=True)

        if self.opt.load_weights_folder != "":
            self.load_model()

        print("Using split:\n  ", self.opt.split)
        print(
            "There are {:d} training items and {:d} validation items\n".format(
                len(train_dataset),
                len(val_dataset)))
    
  
    def train(self):
        if not os.path.isdir(self.opt.log_root):
            os.mkdir(self.opt.log_root)

        for self.epoch in range(self.start_epoch, self.opt.num_epochs + 1):
            self.adjust_learning_rate(self.model_optimizer, self.epoch, self.opt.lr_steps)
            loss = self.run_epoch()
            output = ("Epoch: %d | lr:%.7f | Loss: %.4f | topview Loss: %.4f | transform_topview Loss: %.4f | "
                      "transform Loss: %.4f"
                      % (self.epoch, self.model_optimizer.param_groups[-1]['lr'], loss["loss"], loss["topview_loss"],
                         loss["transform_topview_loss"], loss["transform_loss"]))
            print(output)
            self.log.write(output + '\n')
            self.log.flush()
            for loss_name in loss:
                self.writer.add_scalar(loss_name + '/train', loss[loss_name], global_step=self.epoch)

            if self.epoch % self.opt.log_frequency == 0:
                self.validation(self.log)
                if self.opt.model_split_save:
                    self.save_model()
        self.save_model()

    def process_batch(self, inputs, validation=False):
        outputs = {}
        for key, input in inputs.items():
            if key != "filename":
                inputs[key] = input.to(self.device)

        # x = inputs["color"]
        x = torch.cat([inputs["color"], inputs["depth_gt"]], dim=1)

        outputs = {}
        outputs["topview"] = self.pipeline(x)

        losses = self.criterion(self.opt, self.weight, inputs, outputs)

        # features = self.models["encoder"](inputs["color"])

        # b, c, h, w = features.shape
        # features = (features.reshape(b, c, -1) + self.pos_emb1D[:, :, :h*w].to(self.device)).reshape(b, c, h, w)
        
        
        # x_feature = features
        # transform_feature, retransform_features = self.models["CycledViewProjection"](features)
        # features = self.models["CrossViewTransformer"](features, transform_feature, retransform_features)

        # depth_features = self.models["DepthEncoder"](inputs["depth_gt"])
        # features = self.models["MergeMultimodal"](features,  depth_features)
        
        # chandrakar_features = self.models["ChandrakarEncoder"](inputs["chandrakar_input"])
        # features = self.models["MergeMultimodal"](features,  chandrakar_features)
        # x_feature = retransform_features = transform_feature = features #= depth_features
        # features = self.models["BasicTransformer"](features)

        # chandrakar_features = self.models["ChandrakarEncoder"](inputs["chandrakar_input"])
        # features = self.models["BasicTransformer2"](chandrakar_features, features, chandrakar_features)
        # x_feature = retransform_features = transform_feature = features

        # x_feature = retransform_features = transform_feature = features #= depth_features
        # features = self.models["BasicTransformer"](features, features, features)

        # depth = F.interpolate(input=inputs["depth_gt"], size=(w, h), mode='bilinear')
        # pc = depth.reshape(b, 1, -1) * self.cam_coords.to(self.device).unsqueeze(dim=0).repeat(b, 1, 1)
        # pc = pc.transpose(-1, -2)
        # pc[..., 1] += self.opt.cam_height

        # map_size = self.opt.occ_map_size // 4
        # cell_size = 3.2/map_size

        # x_indices = (pc[..., 0]//cell_size).reshape(-1).long() + map_size//2
        # y_indices = pc[..., 1].reshape(-1)
        # z_indices = (pc[..., 2]//cell_size).reshape(-1).long() + map_size
        # batch_indices = torch.cat([torch.full([pc.shape[1]], ix, device=x_indices.device, dtype=torch.long) for ix in range(pc.shape[0])])

        # valid_indices = (x_indices >= 0) & (x_indices < map_size) & (z_indices >= 0) & (z_indices < map_size) & (y_indices < self.opt.obstacle_height)
        # flat_idx = ((batch_indices * map_size * map_size) + (z_indices * map_size) + x_indices)[valid_indices]

        # rank = torch.argsort(flat_idx)
        # flat_idx = flat_idx[rank]

        # kept = torch.ones_like(flat_idx, device=flat_idx.device, dtype=torch.long)
        # kept[:-1] = flat_idx[1:] != flat_idx[:-1]

        # b, c, h, w = features.shape
        # features = features.reshape((b, c, -1)).transpose(-2, -1).reshape((-1, c))
        # features = features[valid_indices][rank]

        # feature_sum = torch.cumsum(features, dim=0)
        # x_sum = feature_sum[kept] - features[kept]
        # x_sum = torch.cat([x_sum[:1], x_sum[1:] - x_sum[:-1]], dim=0)

        # warped_feature_grid = torch.zeros((b * map_size * map_size, c), dtype=torch.float32, device=x_sum.device, requires_grad=True).clone()
        # warped_feature_grid[flat_idx] = x_sum
        # warped_feature_grid = warped_feature_grid.reshape((b, map_size, map_size, c)).permute((0, 3, 1, 2)) # B x C x Bh x Bw

        # patch_size = 8
        # chandrakar_patches = rearrange(inputs["chandrakar_input"], "b c (x p_x) (y p_y) -> b (p_x p_y c) x y", p_x=patch_size, p_y=patch_size)

        # chandrakar_features = self.models["ChandrakarEncoder"](inputs["chandrakar_input"])
        # # chandrakar_features = self.models["ChandrakarAttn"](chandrakar_features, chandrakar_features, chandrakar_features)
        # features = self.models["MergeMultimodal"](warped_feature_grid,  chandrakar_features)
        

        # if self.multimodal_input:
        #     chandrakar_features = self.models["ChandrakarEncoder"](inputs["chandrakar_input"])
        #     # features = self.models["MergeMultimodal"](features,  chandrakar_features)

        #     # Cross-view Transformation Module
        #     x_feature = features
        #     transform_feature, retransform_features = self.models["CycledViewProjection"](features)
        #     features = self.models["CrossViewTransformer"](features, transform_feature, retransform_features)

        # if self.multimodal_input:
        #     chandrakar_features = self.models["ChandrakarEncoder"](inputs["chandrakar_input"])

        #     # Cross-view Transformation Module
        #     x_feature = features
        #     transform_feature, retransform_features = self.models["CycledViewProjectionMultimodal"](features, chandrakar_features)
        #     features = self.models["CrossViewTransformer"](features, transform_feature, retransform_features)

        # chandrakar_features = self.models["ChandrakarEncoder"](inputs["chandrakar_input"])
        # features = self.models["MergeMultimodal"](features,  chandrakar_features)

        # x_feature = retransform_features = transform_feature = features #= depth_features
        # chandrakar_features = self.models["BasicTransformer2"](features, features, chandrakar_features)  # Based on RGB similarity, warp chandrakar to entire image.

        # features = torch.cat([features, chandrakar_features], dim=1)
        
        # outputs["topview"] = self.models["decoder"](features)
        # outputs["transform_topview"] = self.models["transform_decoder"](transform_feature)
        # if validation:
        #     return outputs
        # losses = self.criterion(self.opt, self.weight, inputs, outputs, x_feature, retransform_features)

        return outputs, losses

    def run_epoch(self):
        self.model_optimizer.step()
        loss = {
            "loss": 0.0,
            "topview_loss": 0.0,
            "transform_loss": 0.0,
            "transform_topview_loss": 0.0,
            "boundary": 0.0,
            "loss_discr": 0.0
        }
        accumulation_steps = 8
        valid_batches = 0
        for batch_idx, inputs in tqdm.tqdm(enumerate(self.train_loader)):
            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()

            if torch.isnan(losses["loss"]) or torch.isinf(losses["loss"]):
                self.log.write("NaN loss at Epoch: %d Batch_idx: %d Filenames: %s \n" % (self.epoch, batch_idx, ",".join(inputs["filename"])))
                self.log.flush()
                continue

            valid_batches += 1

            losses["loss"] = losses["loss"] / accumulation_steps
            losses["loss"].backward()

            # if ((batch_idx + 1) % accumulation_steps) == 0:
            self.model_optimizer.step()
            #     self.model_optimizer.zero_grad()

            for loss_name in losses:
                loss[loss_name] += losses[loss_name].item()
        # self.scheduler.step()
        for loss_name in loss:
            loss[loss_name] /= valid_batches

        if len(self.train_loader) > valid_batches:
            print("NaN loss encountered in {} batches".format(len(self.train_loader) - valid_batches))
            
        return loss

    def parse_log_data(self, val):
        if isinstance(val, dict):
            out = {}
            for k, v in val.items():
                out[k] = self.parse_log_data(v)
            return out
        elif isinstance(val, torch.Tensor):
            return val.cpu().detach().item()
        else:
            return val

    def validation(self, log):
        iou, mAP = np.array([0.] * self.opt.num_class), np.array([0.] * self.opt.num_class)
        # trans_iou, trans_mAP = np.array([0.] * self.opt.num_class), np.array([0.] * self.opt.num_class)

        # Store scalars as pkl
        step_info = {}

        loss = {
            "loss": 0.0,
            "topview_loss": 0.0,
            "transform_loss": 0.0,
            "transform_topview_loss": 0.0,
            "boundary": 0.0,
            "loss_discr": 0.0
        }
        for batch_idx, inputs in tqdm.tqdm(enumerate(self.val_loader)):
            with torch.no_grad():
                outputs, losses = self.process_batch(inputs, False)

            for loss_name in losses:
                loss[loss_name] += losses[loss_name].item()

            pred = np.squeeze(
                torch.argmax(
                    outputs["topview"][:1].detach(),
                    1).cpu().numpy())
            true = np.squeeze(
                inputs[self.opt.type + "_gt"][:1].detach().cpu().numpy())
            iou += mean_IU(pred, true, self.opt.num_class)
            mAP += mean_precision(pred, true, self.opt.num_class)

            if batch_idx >= 4:
                continue
            
            # COLOR data
            color = invnormalize_imagenet(inputs["color"][0].detach().cpu())
            color = cv2.resize(color.numpy().transpose((1,2,0)), dsize=(128, 128)).transpose((2, 0, 1))
            self.writer.add_image(f"color_gt/{batch_idx}", color, self.epoch)

            # DEPTH data
            depth = inputs["depth_gt"][0].detach().cpu().squeeze(dim=0).numpy()
            depth = np.expand_dims(cv2.resize(depth, dsize=(128, 128)), axis=0)
            depth = (depth * 1.14) + 1.67
            self.writer.add_image(f"depth/{batch_idx}", normalize_image(depth), self.epoch)

            # BEV data
            self.writer.add_image(f"bev_gt/{batch_idx}",
                normalize_image(np.expand_dims(true, axis=0), (0, 2)), self.epoch)

            # BEV data
            self.writer.add_image(f"bev_pred/{batch_idx}",
                normalize_image(np.expand_dims(pred, axis=0), (0, 2)), self.epoch)

            if self.multimodal_input:            
                # Chandrakar input data
                chandrakar_input = inputs["chandrakar_input"][0].detach().cpu()
                # For chandrakar depth input
                # chandrakar_input = np.expand_dims(cv2.resize(chandrakar_input.numpy().transpose((1,2,0))[..., 1:], dsize=(128, 128)), axis=0) # Taking the second channel only for depth
                # chandrakar_input = (chandrakar_input * 1.14) + 1.67
                
                self.writer.add_image(f"chandrakar_input/{batch_idx}", chandrakar_input, self.epoch)

            if "semantics_gt" in inputs:
                semantics = inputs["semantics_gt"][0].detach().cpu()
                semantics = cv2.resize(semantics.numpy().transpose((1,2,0)), dsize=(128, 128), interpolation=cv2.INTER_NEAREST)

                overlay = np.copy(color) * 0.5
                overlay[0, semantics==0] *= 2
                overlay[1, semantics==1] *= 2
                self.writer.add_image(f"semantics_gt/{batch_idx}", overlay, self.epoch)
        
        for loss_name in loss:
            loss[loss_name] /= len(self.val_loader)
            self.writer.add_scalar(loss_name + '/val', loss[loss_name], global_step=self.epoch)
            step_info[loss_name] = self.parse_log_data(loss[loss_name])

        iou /= len(self.val_loader)
        mAP /= len(self.val_loader)

        mAP_cls =  dict(unknown=mAP[0], occupied=mAP[1], free=mAP[2])
        mIOU_cls = dict(unknown=iou[0], occupied=iou[1], free=iou[2])
        
        self.writer.add_scalars("mAP_cls", mAP_cls, self.epoch)
        self.writer.add_scalars("mIOU_cls", mIOU_cls, self.epoch)

        step_info["mAP_cls"] = self.parse_log_data(mAP_cls)
        step_info["mIOU_cls"] = self.parse_log_data(mIOU_cls)
        step_info["epoch"] = self.epoch
        step_info['learning_rate'] = {
            'base': self.parse_log_data(self.model_optimizer.param_groups[1]['lr']), 
            'lr/transform': self.parse_log_data(self.model_optimizer.param_groups[0]['lr'])
        }

        step_logpath = os.path.join(self.opt.log_root, self.opt.model_name, 'val', 'step_logs', f'{self.epoch}.pkl')
        os.makedirs(os.path.dirname(step_logpath), exist_ok=True)
        with open(step_logpath, 'wb') as f:
            pickle.dump(step_info, f)

        with np.printoptions(precision=4, suppress=True):
            output = "Epoch: {} | Validation: Loss: {} mIOU: {} mAP: {}".format(self.epoch, loss["loss"], iou, mAP)
            print(output)
            log.write(output + '\n')
            log.flush()

    def save_model(self):
        save_path = os.path.join(
            self.opt.save_path,
            self.opt.model_name,
            "weights_{}".format(
                self.epoch)
        )

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # for model_name, model in self.models.items():
        #     model_path = os.path.join(save_path, "{}.pth".format(model_name))
        #     state_dict = model.state_dict()
        #     state_dict['epoch'] = self.epoch
        #     if model_name == "encoder":
        #         state_dict["height"] = self.opt.height
        #         state_dict["width"] = self.opt.width

        #     torch.save(state_dict, model_path)

        pipeline_state_dict = self.pipeline.state_dict()
        pipeline_state_dict["class"] = type(self.pipeline).__name__
        torch.save(pipeline_state_dict, os.path.join(save_path, "pipeline.pth"))

        optim_path = os.path.join(save_path, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), optim_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(
            self.opt.load_weights_folder)

        if self.opt.load_weights_folder == 'None':
            print("No weights loaded.")
            return

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print(
            "loading model from folder {}".format(
                self.opt.load_weights_folder))

        for key in self.models.keys():
            if "discriminator" not in key:
                print("Loading {} weights...".format(key))
                path = os.path.join(
                    self.opt.load_weights_folder,
                    "{}.pth".format(key))
                model_dict = self.models[key].state_dict()
                pretrained_dict = torch.load(path)
                if 'epoch' in pretrained_dict:
                    self.start_epoch = pretrained_dict['epoch']
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[key].load_state_dict(model_dict)

        # Pipeline param load
        path = os.path.join(self.opt.load_weights_folder, "pipeline.pth")
        if os.path.exists(path):
            model_dict = self.pipeline.state_dict()
            pretrained_dict = torch.load(path)
            print("LOADING PIPELINE WEIGHTS FOR CLASS: ", pretrained_dict["class"])
            if 'epoch' in pretrained_dict:
                self.start_epoch = pretrained_dict['epoch']
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.pipeline.load_state_dict(model_dict)

        # loading adam state
        if self.opt.load_weights_folder == "":
            optimizer_load_path = os.path.join(
                self.opt.load_weights_folder, "adam.pth")
            if os.path.isfile(optimizer_load_path):
                print("Loading Adam weights")
                optimizer_dict = torch.load(optimizer_load_path)
                self.model_optimizer.load_state_dict(optimizer_dict)
            else:
                print("Cannot find Adam weights so Adam is randomly initialized")

    def adjust_learning_rate(self, optimizer, epoch, lr_steps):
        """Sets the learning rate to the initial LR decayed by 10 every 25 epochs"""
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        decay = round(decay, 2)
        lr = self.opt.lr * decay
        lr_transform = self.opt.lr_transform * decay
        decay = self.opt.weight_decay
        optimizer.param_groups[0]['lr'] = lr_transform
        optimizer.param_groups[1]['lr'] = lr
        optimizer.param_groups[0]['weight_decay'] = decay
        optimizer.param_groups[1]['weight_decay'] = decay
        self.writer.add_scalar('lr/base', lr, self.epoch)
        self.writer.add_scalar('lr/transform', lr_transform, self.epoch)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr
        #     param_group['lr'] = lr_transform
        # param_group['weight_decay'] = decay

    def set_seed(self):
        seed = self.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    start_time = time.ctime()
    print(start_time)
    trainer = Trainer()
    trainer.train()
    end_time = time.ctime()
    print(end_time)
