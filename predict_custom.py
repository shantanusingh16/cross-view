import argparse
import glob
import os

import PIL.Image as pil

import cv2

from crossView import model, CrossViewTransformer, CycledViewProjection

import numpy as np
import time
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

from torchvision import transforms

import albumentations as A
import albumentations.pytorch as Apt

COLORMAP_TOPVIEW = np.array([[0, 0, 0], [127, 127, 127], [254, 254, 254]], dtype=np.uint8)

imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

IMG_SIZE = 512


def get_pix_coords(height, width):
    meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = torch.from_numpy(id_coords)
    ones = torch.ones(1, 1, height * width)

    pix_coords = torch.unsqueeze(torch.stack(
        [id_coords[0].view(-1), id_coords[1].view(-1)], 0), 0)
    pix_coords = torch.cat([pix_coords, ones], 1)

    return pix_coords.reshape(1,3,height, width)


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
    device = torch.device("cuda")

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
    normalize = A.Compose([
        A.Normalize(**imagenet_stats, always_apply=True),
        Apt.transforms.ToTensorV2(always_apply=True)
    ])

    crop = A.Compose([
        A.augmentations.geometric.resize.SmallestMaxSize(max_size=IMG_SIZE, interpolation=cv2.INTER_CUBIC, always_apply=True),
        A.augmentations.crops.transforms.CenterCrop(IMG_SIZE, IMG_SIZE, always_apply=True),
    ])


    test_fps = [os.path.join(args.data_path, x) for x in os.listdir(args.data_path) if os.path.splitext(x)[1] == '.jpg']

    print("-> Predicting on {:d} test images".format(len(test_fps)))

    iou, mAP = np.array([0.] * args.num_class), np.array([0.] * args.num_class)

    start_time = time.time()

    print('#Params:', sum(p.numel() for p in pipeline.parameters()))

    ## Correct for camera intrinsics
    K_src = np.array([378.47052001953125, 0.0, 318.99395751953125, 
        0.0, 378.2340087890625, 253.28782653808594, 
        0.0, 0.0, 1.0]).reshape((3,3)) 
    K_src = torch.from_numpy(K_src).float().unsqueeze(dim=0).to(device)

    K_tgt = np.array([256, 0, 256,
        0, 256, 256,
        0, 0, 1]).reshape((3,3))
    invK_tgt = torch.from_numpy(np.linalg.inv(K_tgt)).float().unsqueeze(dim=0).to(device)

    pix_coords = get_pix_coords(512, 512).to(device)
    pix_coords = torch.matmul(torch.matmul(K_src, invK_tgt), pix_coords.reshape((1, 3, -1))).reshape((1, 3, 512, 512))
    pix_coords = pix_coords.permute((0, 2, 3, 1))[..., :2]
    pix_coords[..., 0] /= 640
    pix_coords[..., 1] /= 480
    pix_coords = (pix_coords - 0.5) * 2

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for fp in tqdm(test_fps):

            rgb_in = cv2.cvtColor(cv2.imread(fp, -1), cv2.COLOR_BGR2RGB)

            # rgb_in = torch.from_numpy(rgb_in).unsqueeze(0).permute((0,3,1,2)).float().to(device)

            # # Correct for camera intrinsics
            # rgb = F.grid_sample(rgb_in,
            #             pix_coords,
            #             padding_mode="zeros")
            # rgb = rgb.squeeze().cpu().numpy().transpose(1,2,0)

            
            rgb_cropped = crop(image=rgb_in)['image']
            rgb_src = normalize(image=rgb_cropped)['image'].float().unsqueeze(0).to(device)

            # PREDICTION
            
            outputs = {}
            tv = pipeline.forward(rgb_src)
            # h_tv, w_tv =  tv.shape[2:]
            
            # attn_map = pipeline.get_attention_map().cpu().detach().mean(dim=1)
            # residual_attn = torch.eye(attn_map.shape[1])
            # attn_map = attn_map + residual_attn
            # attn_map = attn_map / attn_map.sum(-1).unsqueeze(-1)
            # # # attn_map = F.interpolate(attn_map.unsqueeze(1), size=(h_tv * w_tv, h_tv * w_tv)).squeeze(1)
            
            # grid_size = int(np.sqrt(attn_map.size(-1)))
            
            # tv_scaled = torch.nn.Softmax(dim=1)(F.interpolate(tv, size=(grid_size, grid_size)))
            # pred_scaled = torch.argmax(tv_scaled.detach(), dim=1)
            
            # occ = attn_map.reshape(-1, attn_map.shape[-1])[pred_scaled.reshape(-1) == 1, :]
            # occ = torch.mean(occ, dim=0).reshape(grid_size, grid_size)
            # # occ = (torch.sum(occ, dim=0)/occ.shape[0]).reshape(grid_size, grid_size)
            # occ = occ.cpu().detach()
            
            # free = attn_map.reshape(-1, attn_map.shape[-1])[pred_scaled.reshape(-1) == 2, :]
            # free = torch.mean(free, dim=0).reshape(grid_size, grid_size)
            # # free = (torch.sum(free, dim=0)/free.shape[0]).reshape(grid_size, grid_size)
            # free = free.cpu().detach()

            pred = torch.argmax(tv.detach(), 1).cpu().squeeze(dim=0)
            # trues = inputs[args.type + "_gt"].detach().cpu().numpy()

            # if args.grad_cam == True:
            #     with torch.enable_grad():
            #         wrapped_model = SegmentationModelOutputWrapper(pipeline)
            #         target_layers = pipeline.bottleneck
            #         for sem_idx, sem_class in enumerate(["unknown", "occupied", "free"]):
            #             with GradCAM(model=wrapped_model,
            #                         target_layers=target_layers,
            #                         use_cuda=torch.cuda.is_available()) as cam:
            #                 grayscale_cam = cam(input_tensor=rgb, targets=[SemanticSegmentationTarget(sem_idx)])
            #                 rgb_cam = []
            #                 for idx in range(grayscale_cam.shape[0]):
            #                     # img = torch.clamp(invnormalize_imagenet(rgb[idx]), min=0, max=1).cpu().detach().numpy().transpose(1,2,0)
            #                     img = COLORMAP_TOPVIEW[preds[idx].reshape(-1)].reshape((128, 128, 3))
            #                     img = cv2.resize(img, (rgb.shape[2:]), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255
            #                     # rgb_cam.append(img)
            #                     rgb_cam.append(show_cam_on_image(img, grayscale_cam[idx], use_rgb=True))
                            
            #                 outputs[("rgb_cam", sem_class, 0)] = np.array(rgb_cam).transpose(0, 3, 1, 2)
            #                 outputs[("grayscale_cam", sem_class, 0)] = np.expand_dims(grayscale_cam, axis=1)


            # SAVE OUTPUT

            filename = os.path.splitext(os.path.basename(fp))[0].replace("frame", "")
            output_path = os.path.join(args.out_dir, model_name, 'pred', f"{filename}.png")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            true_top_view = pred.numpy().astype(np.uint8) * 127
            cv2.imwrite(output_path, true_top_view)


            # # Raw prob
            # outpath = os.path.join(args.out_dir, model_name, folder, camera_pose, 'prob', f'{fileidx}.npy')
            # os.makedirs(os.path.dirname(outpath), exist_ok=True)
            # np.save(outpath, tv.detach().cpu().numpy()[idx])
            
            # # Log Attention maps
            # bev_dir = 'attention'
            # rgb = invnormalize_imagenet(inputs["color"].squeeze(0)).permute(1,2,0).cpu().detach()
            # rgb = np.clip((rgb).numpy(), a_min=0, a_max=1)

            outpath = os.path.join(args.out_dir, model_name, 'pred_combined', f'{filename}.png')
            os.makedirs(os.path.dirname(outpath), exist_ok=True)

            pred_scaled = F.interpolate(torch.argmax(tv.detach(), dim=1, keepdim=True).float(), size=(IMG_SIZE, IMG_SIZE), mode='nearest').squeeze().long().cpu().numpy()
            pred_colored = COLORMAP_TOPVIEW[pred_scaled.reshape(-1)].reshape((IMG_SIZE, IMG_SIZE, 3))

            combined = np.zeros((IMG_SIZE, IMG_SIZE*2 + 10, 3), dtype=np.uint8)
            combined[:, :IMG_SIZE, :] = rgb_cropped
            combined[:, IMG_SIZE+10:, :] = pred_colored #.permute(1,2,0)
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

            # if (("rgb_cam", "unknown", 0) in outputs):
            #     for sem_class in ["unknown", "occupied", "free"]:
            #         bev_dir = 'grad_rgb'
            #         img = outputs[("rgb_cam", sem_class, 0)][idx].transpose(1,2,0)
            #         outpath = os.path.join(args.out_dir, bev_dir, '{}_{}.png'.format(sem_class, filename))
            #         os.makedirs(os.path.dirname(outpath), exist_ok=True)
            #         cv2.imwrite(outpath, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            #         bev_dir = 'grad_grayscale'
            #         img = outputs[("grayscale_cam", sem_class, 0)][idx].transpose(1,2,0)
            #         outpath = os.path.join(args.out_dir, model_name,bev_dir, '{}_{}.png'.format(sem_class, filename))
            #         img = (img.squeeze() * 255).astype(np.uint8)
            #         os.makedirs(os.path.dirname(outpath), exist_ok=True)
            #         cv2.imwrite(outpath, img)

    print('FPS:',  1000/(time.time() - start_time))

    iou = iou / len(test_fps)
    mAP = mAP / len(test_fps)
    with np.printoptions(precision=4, suppress=True):
        print('mIOU:', iou)
        print('mAP:', mAP)
    print('-> Done!')


if __name__ == "__main__":
    args = get_args()
    test(args)
