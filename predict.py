import argparse
import glob
import os

import PIL.Image as pil

import cv2

from crossView import model, CrossViewTransformer, CycledViewProjection

import numpy as np

import torch
from torch.utils.data import DataLoader

from easydict import EasyDict as edict

import crossView

from tqdm import tqdm
from utils import mean_IU, mean_precision


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    if os.path.islink(filename):
        filename = os.readlink(filename)
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def get_args():
    parser = argparse.ArgumentParser(
        description="Testing options")
    parser.add_argument("--data_path", type=str, help="Path to the root data directory")
    parser.add_argument("--bev_dir", type=str, help="Path to the bev directory (Only for habitat dataset)")
    parser.add_argument("--semantics_dir", type=str, help="Path to the semantics directory (Only for habitat dataset)")
    parser.add_argument("--chandrakar_input_dir", type=str, help="Path to the chandrakar input directory (Only for habitat dataset)")
    parser.add_argument("--floor_path", type=str, help="Path to the floor maps directory (Only for habitat dataset)")
    parser.add_argument("--model_path", type=str,
                        help="path to model weights", required=True)
    parser.add_argument(
        "--ext",
        type=str,
        default="png",
        help="extension of images in the folder")
    parser.add_argument("--height", type=int, default=1024,
                        help="Image height")
    parser.add_argument("--width", type=int, default=1024,
                        help="Image width")
    parser.add_argument("--out_dir", type=str,
                        default="output directory to save topviews")
    parser.add_argument("--type", type=str,
                        default="static/dynamic/both")
    # parser.add_argument("--view", type=str, default=1, help="view number")
    parser.add_argument(
        "--split",
        type=str,
        choices=[
            "argo",
            "3Dobject",
            "odometry",
            "raw",
            "gibson",
            "gibson4"],
        help="Data split for training/validation")
    parser.add_argument("--occ_map_size", type=int, default=256,
                        help="size of topview occupancy map")
    parser.add_argument("--num_class", type=int, default=2,
                        help="Number of classes")
    parser.add_argument("--batch_size", type=int, default=6,
                        help="Mini-Batch size")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of cpu workers for dataloaders")

    configs = edict(vars(parser.parse_args()))
    return configs


def test(args):
    models = {}
    device = torch.device("cuda")
    encoder_path = os.path.join(args.model_path, "encoder.pth")
    encoder_dict = torch.load(encoder_path, map_location=device)
    feed_height = encoder_dict["height"]
    feed_width = encoder_dict["width"]
    models["encoder"] = model.Encoder(18, feed_width, feed_height, False)
    filtered_dict_enc = {
        k: v for k,
        v in encoder_dict.items() if k in models["encoder"].state_dict()}
    models["encoder"].load_state_dict(filtered_dict_enc)

    CVP_path = os.path.join(args.model_path, "CycledViewProjection.pth")
    CVP_dict = torch.load(CVP_path, map_location=device)
    models['CycledViewProjection'] = CycledViewProjection(in_dim=8)
    filtered_dict_cvp = {
        k: v for k,
        v in CVP_dict.items() if k in models["CycledViewProjection"].state_dict()}
    models["CycledViewProjection"].load_state_dict(filtered_dict_cvp)

    CVT_path = os.path.join(args.model_path, "CrossViewTransformer.pth")
    CVT_dict = torch.load(CVT_path, map_location=device)
    models['CrossViewTransformer'] = CrossViewTransformer(128)
    filtered_dict_cvt = {
        k: v for k,
        v in CVT_dict.items() if k in models["CrossViewTransformer"].state_dict()}
    models["CrossViewTransformer"].load_state_dict(filtered_dict_cvt)

    decoder_path = os.path.join(args.model_path, "decoder.pth")
    DEC_dict = torch.load(decoder_path, map_location=device)
    models["decoder"] = model.Decoder(
        models["encoder"].resnet_encoder.num_ch_enc, args.num_class, args.occ_map_size)
    filtered_dict_dec = {
        k: v for k,
        v in DEC_dict.items() if k in models["decoder"].state_dict()}
    models["decoder"].load_state_dict(filtered_dict_dec)

    transform_decoder_path = os.path.join(args.model_path, "transform_decoder.pth")
    TRDEC_dict = torch.load(transform_decoder_path, map_location=device)
    models["transform_decoder"] = model.Decoder(
        models["encoder"].resnet_encoder.num_ch_enc, args.num_class, args.occ_map_size)
    filtered_dict_trdec = {
        k: v for k,
        v in TRDEC_dict.items() if k in models["transform_decoder"].state_dict()}
    models["transform_decoder"].load_state_dict(filtered_dict_trdec)

    chandrakar_encoder_path = os.path.join(args.model_path, "ChandrakarEncoder.pth")
    if os.path.exists(chandrakar_encoder_path):
        CKENC_dict = torch.load(chandrakar_encoder_path, map_location=device)
        models["ChandrakarEncoder"] = crossView.ChandrakarEncoder(2, [4,4,4,2], 16)
        # models["ChandrakarEncoder"] = crossView.ChandrakarEncoder(1, [2,2,2,2], 16)
        filtered_dict_ckenc = {
            k: v for k,
            v in CKENC_dict.items() if k in models["ChandrakarEncoder"].state_dict()}
        models["ChandrakarEncoder"].load_state_dict(filtered_dict_ckenc)

    merge_multimodal_path = os.path.join(args.model_path, "MergeMultimodal.pth")
    if os.path.exists(merge_multimodal_path):
        merge_multimodal_dict = torch.load(merge_multimodal_path, map_location=device)
        models["MergeMultimodal"] = crossView.MergeMultimodal(128, 2)
        filtered_dict_merge_multimodal = {
            k: v for k,
            v in merge_multimodal_dict.items() if k in models["MergeMultimodal"].state_dict()}
        models["MergeMultimodal"].load_state_dict(filtered_dict_merge_multimodal)

    cvp_multimodal_path = os.path.join(args.model_path, "CycledViewProjectionMultimodal.pth")
    if os.path.exists(cvp_multimodal_path):
        cvp_multimodal_dict = torch.load(cvp_multimodal_path, map_location=device)
        models['CycledViewProjectionMultimodal'] = crossView.CycledViewProjectionMultimodal(in_dim=8, in_channels=128)
        filtered_dict_cvp_multimodal = {
            k: v for k,
            v in cvp_multimodal_dict.items() if k in models["CycledViewProjectionMultimodal"].state_dict()}
        models["CycledViewProjectionMultimodal"].load_state_dict(filtered_dict_cvp_multimodal)

    for key in models.keys():
        models[key].to(device)
        models[key].eval()

    model_name = os.path.basename(os.path.dirname(os.path.dirname(args.model_path)))

    dataset_dict = {
        "3Dobject": crossView.KITTIObject,
        "odometry": crossView.KITTIOdometry,
        "argo": crossView.Argoverse,
        "raw": crossView.KITTIRAW,
        "gibson": crossView.GibsonDataset,
        "gibson4": crossView.Gibson4Dataset
    }

    dataset = dataset_dict[args.split]
    fpath = os.path.join(
            os.path.dirname(__file__),
            "splits",
            args.split,
            "val_files.txt")

    test_filenames = readlines(fpath)

    val_dataset = dataset(args, test_filenames, is_train=False)

    val_loader = DataLoader(
            val_dataset,
            args.batch_size,
            True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True)

    print("-> Predicting on {:d} test images".format(len(val_dataset)))

    iou, mAP = np.array([0.] * args.num_class), np.array([0.] * args.num_class)

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for inputs in tqdm(val_loader):
            for key, input in inputs.items():
                if key != "filename":
                    inputs[key] = input.to(device)

            # PREDICTION
            features = models["encoder"](inputs["color"])
        
            # transform_feature, retransform_features = models["CycledViewProjection"](features)
            # features = models["CrossViewTransformer"](features, transform_feature, retransform_features)

            # x_feature = retransform_features = transform_feature = features
            features = models["BasicTransformer"](features)

            ###  MERGE CHANDRAKAR DEPTH AND RGB FEATURES
            chandrakar_features = models["ChandrakarEncoder"](inputs["chandrakar_input"])
            # features = models["MergeMultimodal"](features,  chandrakar_features)

            # Cross-view Transformation Module
            # transform_feature, retransform_features = models["CycledViewProjection"](features)
            # features = models["CrossViewTransformer"](features, transform_feature, retransform_features)

            ### MERGE CHANDRAKAR BEV AND PRETRANSFORMED BEV FEATURES
            # chandrakar_features = self.models["ChandrakarEncoder"](inputs["chandrakar_input"])

            # # Cross-view Transformation Module
            # x_feature = features
            # transform_feature, retransform_features = self.models["CycledViewProjectionMultimodal"](features, chandrakar_features)
            # features = self.models["CrossViewTransformer"](features, transform_feature, retransform_features)


            ### MERGE POST TRANSFORMED BEV FEATURES AND CHANDRAKAR BEV
            # features = models["MergeMultimodal"](features,  chandrakar_features)

            tv = models["decoder"](features)
            # outputs["transform_topview"] = self.models["transform_decoder"](transform_feature)

            preds = np.squeeze(torch.argmax(tv.detach(), 1).cpu().numpy())
            trues = np.squeeze(inputs[args.type + "_gt"].detach().cpu().numpy())

            # SAVE OUTPUT
            for idx in range(args.batch_size):
                pred = preds[idx]
                true = trues[idx]
                iou += mean_IU(pred, true, args.num_class)
                mAP += mean_precision(pred, true, args.num_class)

                folder, fileidx = inputs["filename"][idx].split()
                output_path = os.path.join(args.out_dir, model_name, folder, "{}.png".format(fileidx))
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                true_top_view = pred.astype(np.uint8) * 127
                cv2.imwrite(output_path, true_top_view)

    iou = iou / len(val_dataset)
    mAP = mAP / len(val_dataset)
    with np.printoptions(precision=4, suppress=True):
        print('mIOU:', iou)
        print('mAP:', mAP)
    print('-> Done!')


if __name__ == "__main__":
    args = get_args()
    test(args)
