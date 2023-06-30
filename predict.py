import argparse
import glob
import os

import PIL.Image as pil

import cv2

from crossView import model, CrossViewTransformer, CycledViewProjection

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from easydict import EasyDict as edict

from utils import invnormalize_imagenet

import crossView

from tqdm import tqdm
from utils import mean_IU, mean_precision, invnormalize_imagenet

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from crossView.pipelines.transformer import MultiBlockTransformer, P_BasicTransformer, BasicTransformer_Old

from crossView.grad_cam import SegmentationModelOutputWrapper, SemanticSegmentationTarget


COLORMAP_TOPVIEW = np.array([[93, 57, 154], [67, 232, 172], [248, 255, 144]], dtype=np.uint8) 

COLORMAP_TOPVIEW = np.array([[0, 0, 0], [127, 127, 127], [255, 255, 255]], dtype=np.uint8)


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
    parser.add_argument('--grad_cam', type=bool, default=False)

    configs = edict(vars(parser.parse_args()))
    return configs


def test(args):
    # models = {}
    device = torch.device("cuda")
    # encoder_path = os.path.join(args.model_path, "encoder.pth")
    # encoder_dict = torch.load(encoder_path, map_location=device)
    # feed_height = encoder_dict["height"]
    # feed_width = encoder_dict["width"]
    # models["encoder"] = model.Encoder(18, feed_width, feed_height, False)
    # filtered_dict_enc = {
    #     k: v for k,
    #     v in encoder_dict.items() if k in models["encoder"].state_dict()}
    # models["encoder"].load_state_dict(filtered_dict_enc)

    # CVP_path = os.path.join(args.model_path, "CycledViewProjection.pth")
    # if os.path.exists(CVP_path):
    #     CVP_dict = torch.load(CVP_path, map_location=device)
    #     models['CycledViewProjection'] = CycledViewProjection(in_dim=8)
    #     filtered_dict_cvp = {
    #         k: v for k,
    #         v in CVP_dict.items() if k in models["CycledViewProjection"].state_dict()}
    #     models["CycledViewProjection"].load_state_dict(filtered_dict_cvp)

    # CVT_path = os.path.join(args.model_path, "CrossViewTransformer.pth")
    # if os.path.exists(CVT_path):
    #     CVT_dict = torch.load(CVT_path, map_location=device)
    #     models['CrossViewTransformer'] = CrossViewTransformer(128)
    #     filtered_dict_cvt = {
    #         k: v for k,
    #         v in CVT_dict.items() if k in models["CrossViewTransformer"].state_dict()}
    #     models["CrossViewTransformer"].load_state_dict(filtered_dict_cvt)

    # decoder_path = os.path.join(args.model_path, "decoder.pth")
    # DEC_dict = torch.load(decoder_path, map_location=device)
    # models["decoder"] = model.Decoder(
    #     models["encoder"].resnet_encoder.num_ch_enc, args.num_class, args.occ_map_size)
    # filtered_dict_dec = {
    #     k: v for k,
    #     v in DEC_dict.items() if k in models["decoder"].state_dict()}
    # models["decoder"].load_state_dict(filtered_dict_dec)

    # transform_decoder_path = os.path.join(args.model_path, "transform_decoder.pth")
    # TRDEC_dict = torch.load(transform_decoder_path, map_location=device)
    # models["transform_decoder"] = model.Decoder(
    #     models["encoder"].resnet_encoder.num_ch_enc, args.num_class, args.occ_map_size, in_features=128)
    # filtered_dict_trdec = {
    #     k: v for k,
    #     v in TRDEC_dict.items() if k in models["transform_decoder"].state_dict()}
    # models["transform_decoder"].load_state_dict(filtered_dict_trdec)

    # base_transformer_path = os.path.join(args.model_path, "BasicTransformer.pth")
    # if os.path.exists(base_transformer_path):
    #     bTR_dict = torch.load(base_transformer_path, map_location=device)
    #     models["BasicTransformer"] = crossView.BasicTransformer(8, 128)
    #     filtered_dict_btr = {
    #         k: v for k,
    #         v in bTR_dict.items() if k in models["BasicTransformer"].state_dict()}
    #     models["BasicTransformer"].load_state_dict(filtered_dict_btr)

    # chandrakar_encoder_path = os.path.join(args.model_path, "ChandrakarEncoder.pth")
    # if os.path.exists(chandrakar_encoder_path):
    #     CKENC_dict = torch.load(chandrakar_encoder_path, map_location=device)
    #     # models["ChandrakarEncoder"] = crossView.ChandrakarEncoder(2, [4,4,4,2], 16)
    #     # models["ChandrakarEncoder"] = crossView.ChandrakarEncoder(1, [2,2,2,2], 16)
    #     models["ChandrakarEncoder"] = crossView.Encoder(18, feed_width, feed_height, False, False, 2)
    #     filtered_dict_ckenc = {
    #         k: v for k,
    #         v in CKENC_dict.items() if k in models["ChandrakarEncoder"].state_dict()}
    #     models["ChandrakarEncoder"].load_state_dict(filtered_dict_ckenc)

    # merge_multimodal_path = os.path.join(args.model_path, "MergeMultimodal.pth")
    # if os.path.exists(merge_multimodal_path):
    #     merge_multimodal_dict = torch.load(merge_multimodal_path, map_location=device)
    #     models["MergeMultimodal"] = crossView.MergeMultimodal(128, 2)
    #     filtered_dict_merge_multimodal = {
    #         k: v for k,
    #         v in merge_multimodal_dict.items() if k in models["MergeMultimodal"].state_dict()}
    #     models["MergeMultimodal"].load_state_dict(filtered_dict_merge_multimodal)

    # cvp_multimodal_path = os.path.join(args.model_path, "CycledViewProjectionMultimodal.pth")
    # if os.path.exists(cvp_multimodal_path):
    #     cvp_multimodal_dict = torch.load(cvp_multimodal_path, map_location=device)
    #     models['CycledViewProjectionMultimodal'] = crossView.CycledViewProjectionMultimodal(in_dim=8, in_channels=128)
    #     filtered_dict_cvp_multimodal = {
    #         k: v for k,
    #         v in cvp_multimodal_dict.items() if k in models["CycledViewProjectionMultimodal"].state_dict()}
    #     models["CycledViewProjectionMultimodal"].load_state_dict(filtered_dict_cvp_multimodal)

    # for key in models.keys():
    #     models[key].to(device)
    #     models[key].eval()
    
    # pipeline = BasicTransformer_Old(models=None, opt=args)
    # for filename in os.listdir(args.model_path):
    #     k = filename.replace(".pth", "")
    #     if hasattr(pipeline, k):
    #         weights = torch.load(os.path.join(args.model_path, filename), map_location=device)
    #         model = getattr(pipeline, k)
    #         print(k, [x for x in model.state_dict() if x not in weights])
    #         filtered_weights = {x: y for x, y in weights.items() if x in model.state_dict()}
    #         mk, uk = model.load_state_dict(filtered_weights)
    #         print(mk, uk)
        
    # pipeline.to(device)
    # pipeline.eval()
    
    # pipeline_state_dict = pipeline.state_dict()
    # pipeline_state_dict["class"] = type(pipeline).__name__
    # torch.save(pipeline_state_dict, os.path.join(args.model_path, "pipeline.pth"))

    pipeline_path = os.path.join(args.model_path, "pipeline.pth")
    if os.path.exists(pipeline_path):
        pipeline_dict = torch.load(pipeline_path, map_location=device)
        print("LOADING PIPELINE WEIGHTS FOR CLASS: ", pipeline_dict["class"])
        pipeline = MultiBlockTransformer(models=None, opt=args, nblocks=6)
        filtered_dict_pipeline = {
            k: v for k,
            v in pipeline_dict.items() if k in pipeline.state_dict()}
        print([k for k in pipeline_dict if k not in pipeline.state_dict()])
        pipeline.load_state_dict(filtered_dict_pipeline)
        pipeline.to(device)
        pipeline.eval()
    else:
        print("PIPELINE NOT LOADED. FIX THE CODE RUN BELOW MANUALLY")

    model_name = os.path.basename(os.path.dirname(args.model_path))

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
            drop_last=False)

    print("-> Predicting on {:d} test images".format(len(val_dataset)))

    iou, mAP = np.array([0.] * args.num_class), np.array([0.] * args.num_class)

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for inputs in tqdm(val_loader):
            for key, input in inputs.items():
                if key != "filename":
                    inputs[key] = input.to(device)

            rgb = inputs["color"]
            # PREDICTION
            # features = models["encoder"](inputs["color"])
        
            # transform_feature, retransform_features = models["CycledViewProjection"](features)
            # features = models["CrossViewTransformer"](features, transform_feature, retransform_features)

            # x_feature = retransform_features = transform_feature = features
            # features = models["BasicTransformer"](features)

            ###  MERGE CHANDRAKAR DEPTH AND RGB FEATURES
            # chandrakar_features = models["ChandrakarEncoder"](inputs["chandrakar_input"])
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

            # tv = models["decoder"](features)
            # outputs["transform_topview"] = self.models["transform_decoder"](transform_feature)
            outputs = {}
            tv = pipeline.forward(inputs["color"]).cpu().detach()
            h_tv, w_tv =  tv.shape[2:]

            B, C, H, W = inputs["color"].shape
            
            attn_map = pipeline.get_attention_map().cpu().detach().mean(dim=1)
            residual_attn = torch.eye(attn_map.shape[1])
            attn_map = attn_map + residual_attn
            attn_map = attn_map / attn_map.sum(-1).unsqueeze(-1)
            # attn_map = F.interpolate(attn_map.unsqueeze(1), size=(h_tv * w_tv, h_tv * w_tv)).squeeze(1)
            
            grid_size = int(np.sqrt(attn_map.size(-1)))
            
            tv_scaled = torch.nn.Softmax(dim=1)(F.interpolate(tv, size=(grid_size, grid_size)))
            # pred_scaled = torch.argmax(tv_scaled.detach(), dim=1)
            tv_flattened = tv_scaled.reshape((*tv_scaled.shape[:2], -1))

            occ_max_idx = torch.argmax(tv_flattened[:, 1], dim=-1)
            occ_tv_map = torch.zeros(attn_map.size(0), grid_size * grid_size, device=attn_map.device, dtype=torch.float)
            occ_tv_map[torch.arange(attn_map.size(0)), occ_max_idx] = 1
            occ_tv_map = F.interpolate(occ_tv_map.reshape(attn_map.size(0), 1, grid_size, grid_size), (h_tv, w_tv), mode='bicubic')
            occ_pers_map = attn_map[torch.arange(attn_map.size(0)), occ_max_idx].reshape(attn_map.size(0), 1, grid_size, grid_size)
            occ_pers_map = F.interpolate(occ_pers_map, (H, W), mode='bicubic')

            free_max_idx = torch.argmax(tv_flattened[:, 2], dim=-1)
            free_tv_map = torch.zeros(attn_map.size(0), grid_size * grid_size, device=attn_map.device, dtype=torch.float)
            free_tv_map[torch.arange(attn_map.size(0)), free_max_idx] = 1
            free_tv_map = F.interpolate(free_tv_map.reshape(attn_map.size(0), 1, grid_size, grid_size), (h_tv, w_tv), mode='bicubic')
            free_pers_map = attn_map[torch.arange(attn_map.size(0)), free_max_idx].reshape(attn_map.size(0), 1, grid_size, grid_size)
            free_pers_map = F.interpolate(free_pers_map, (H, W), mode='bicubic')

            
            # occ = attn_map.reshape(-1, attn_map.shape[-1])[pred_scaled.reshape(-1) == 1, :]
            # occ = torch.mean(occ, dim=0).reshape(grid_size, grid_size)
            # # occ = (torch.sum(occ, dim=0)/occ.shape[0]).reshape(grid_size, grid_size)
            # occ = occ.cpu().detach()
            
            # free = attn_map.reshape(-1, attn_map.shape[-1])[pred_scaled.reshape(-1) == 2, :]
            # free = torch.mean(free, dim=0).reshape(grid_size, grid_size)
            # # free = (torch.sum(free, dim=0)/free.shape[0]).reshape(grid_size, grid_size)
            # free = free.cpu().detach()

            if args.grad_cam == True:
                with torch.enable_grad():
                    wrapped_model = SegmentationModelOutputWrapper(pipeline)
                    target_layers = pipeline.bottleneck
                    for sem_idx, sem_class in enumerate(["unknown", "occupied", "free"]):
                        with GradCAM(model=wrapped_model,
                                    target_layers=target_layers,
                                    use_cuda=torch.cuda.is_available()) as cam:
                            grayscale_cam = cam(input_tensor=rgb, targets=[SemanticSegmentationTarget(sem_idx)])
                            rgb_cam = []
                            for idx in range(grayscale_cam.shape[0]):
                                img = torch.clamp(invnormalize_imagenet(rgb[idx]), min=0, max=1).cpu().detach().numpy().transpose(1,2,0)
                                rgb_cam.append(show_cam_on_image(img, grayscale_cam[idx], use_rgb=True))
                            
                            outputs[("rgb_cam", sem_class, 0)] = np.array(rgb_cam).transpose(0, 3, 1, 2)
                            outputs[("grayscale_cam", sem_class, 0)] = np.expand_dims(grayscale_cam, axis=1)


            preds = torch.argmax(tv.detach(), 1).cpu().numpy()
            trues = inputs[args.type + "_gt"].detach().cpu().numpy()

            # SAVE OUTPUT
            for idx in range(preds.shape[0]):
                pred = preds[idx]
                true = trues[idx]
                iou += mean_IU(pred, true, args.num_class)
                mAP += mean_precision(pred, true, args.num_class)

                folder, camera_pose, fileidx = inputs["filename"][idx].split()
                output_path = os.path.join(args.out_dir, model_name, folder, camera_pose, f"{fileidx}.png")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                true_top_view = pred.astype(np.uint8) * 127
                cv2.imwrite(output_path, true_top_view)

                # # Raw prob
                # outpath = os.path.join(args.out_dir, model_name, folder, camera_pose, 'prob', f'{fileidx}.npy')
                # os.makedirs(os.path.dirname(outpath), exist_ok=True)
                # np.save(outpath, tv.detach().cpu().numpy()[idx])
                
                # Log Attention maps
                bev_dir = 'attention'
                rgb = invnormalize_imagenet(inputs["color"][idx]).permute(1,2,0).cpu().detach()
                rgb = np.clip((rgb).numpy(), a_min=0, a_max=1)

                h, w, c = rgb.shape

                # Log occupied samples
                outpath = os.path.join(args.out_dir, model_name, folder, camera_pose, bev_dir, f'occ_{fileidx}.png')
                os.makedirs(os.path.dirname(outpath), exist_ok=True)
                pers = show_cam_on_image(rgb, occ_pers_map[idx, 0], use_rgb=True)
                color_tv = COLORMAP_TOPVIEW[true_top_view.reshape(-1)//127].reshape((128, 128, 3)).astype(np.float32) / 255
                tv = cv2.resize(show_cam_on_image(color_tv, occ_tv_map[idx, 0]), (h, w), interpolation=cv2.INTER_CUBIC)
                combined = np.zeros((h, w*2 + 10, 3), dtype=np.uint8)
                combined[:, :w, :] = pers
                combined[:, w+10:, :] = tv
                cv2.imwrite(outpath, cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))


                # Log free samples
                outpath = os.path.join(args.out_dir, model_name, folder, camera_pose, bev_dir, f'free_{fileidx}.png')
                os.makedirs(os.path.dirname(outpath), exist_ok=True)
                pers = show_cam_on_image(rgb, free_pers_map[idx, 0], use_rgb=True)
                color_tv = COLORMAP_TOPVIEW[true_top_view.reshape(-1)//127].reshape((128, 128, 3)).astype(np.float32) / 255
                tv = cv2.resize(show_cam_on_image(color_tv, free_tv_map[idx, 0]), (h, w), interpolation=cv2.INTER_CUBIC)
                combined = np.zeros((h, w*2 + 10, 3), dtype=np.uint8)
                combined[:, :w, :] = pers
                combined[:, w+10:, :] = tv
                cv2.imwrite(outpath, cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
                
                # occ = F.interpolate(occ.reshape(1,1,*occ.shape), size=(h, w), mode='bicubic').squeeze(0) #.permute(1,2,0)
                # occ = 1 - (occ - occ.min()) / (occ.max() - occ.min())
                # # occ = occ / torch.sum(occ)
                # occ = show_cam_on_image(rgb, occ.numpy()[0], use_rgb=True)
                
                # free = F.interpolate(free.reshape(1,1,*free.shape), size=(h, w), mode='bicubic').squeeze(0) #.permute(1,2,0)
                # free = 1 - (free - free.min()) / (free.max() - free.min()) 
                # # free = free / torch.sum(free)
                # free = show_cam_on_image(rgb, free.numpy()[0], use_rgb=True)

                # outpath = os.path.join(args.out_dir, model_name, folder, camera_pose, bev_dir, f'{fileidx}.png')
                # os.makedirs(os.path.dirname(outpath), exist_ok=True)

                # combined = np.zeros((h, w*2 + 10, 3), dtype=np.uint8)
                # combined[:, :w, :] = occ
                # combined[:, w+10:, :] = free
                # cv2.imwrite(outpath, cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))

                if (("rgb_cam", "unknown", 0) in outputs):
                    for sem_class in ["unknown", "occupied", "free"]:
                        bev_dir = 'grad_rgb'
                        img = outputs[("rgb_cam", sem_class, 0)][idx].transpose(1,2,0)
                        outpath = os.path.join(args.out_dir, model_name, folder, camera_pose, bev_dir, '{}_{}.png'.format(sem_class, fileidx))
                        os.makedirs(os.path.dirname(outpath), exist_ok=True)
                        cv2.imwrite(outpath, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                        bev_dir = 'grad_grayscale'
                        img = outputs[("grayscale_cam", sem_class, 0)][idx].transpose(1,2,0)
                        outpath = os.path.join(args.out_dir, model_name, folder, camera_pose, bev_dir, '{}_{}.png'.format(sem_class, fileidx))
                        img = (img.squeeze() * 255).astype(np.uint8)
                        os.makedirs(os.path.dirname(outpath), exist_ok=True)
                        cv2.imwrite(outpath, img)

    iou = iou / len(val_dataset)
    mAP = mAP / len(val_dataset)
    with np.printoptions(precision=4, suppress=True):
        print('mIOU:', iou)
        print('mAP:', mAP)
    print('-> Done!')


if __name__ == "__main__":
    args = get_args()
    test(args)
