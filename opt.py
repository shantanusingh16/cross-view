import argparse
import os

from easydict import EasyDict as edict


def get_args():
    parser = argparse.ArgumentParser(description="Training options")
    parser.add_argument("--data_path", type=str, help="Path to the root data directory")
    parser.add_argument("--bev_dir", type=str, help="Path to the bev directory (Only for habitat dataset)")
    parser.add_argument("--semantics_dir", type=str, help="Path to the semantics directory (Only for habitat dataset)")
    parser.add_argument("--chandrakar_input_dir", type=str, help="Path to the chandrakar input directory (Only for habitat dataset)")
    parser.add_argument("--floor_path", type=str, help="Path to the floor maps directory (Only for habitat dataset)")
    parser.add_argument("--save_path", type=str, help="Path to save models")
    
    parser.add_argument(
        "--load_weights_folder",
        type=str,
        default="",
        help="Path to a pretrained model used for initialization")
    parser.add_argument("--model_name", type=str, default="crossView",
                        help="Model Name with specifications")
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
    parser.add_argument("--ext", type=str, default="png",
                        help="File extension of the images")
    parser.add_argument("--height", type=int, default=1024,
                        help="Image height")
    parser.add_argument("--width", type=int, default=1024,
                        help="Image width")
    parser.add_argument(
        "--type",
        type=str,
        choices=[
            "both",
            "static",
            "dynamic"],
        help="Type of model being trained")
    parser.add_argument("--global_seed", type=int, default=0,
                        help="seed")
    parser.add_argument("--batch_size", type=int, default=6,
                        help="Mini-Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,  # attention
                        help="learning rate")
    parser.add_argument("--lr_transform", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--grad_clip_value", type=float, default=None,
                        help="gradient clip value")
    parser.add_argument('--lr_steps', default=50, type=int, nargs="+",  # attention
                        metavar='LRSteps', help='epochs to decay learning rate by 10')                    
    parser.add_argument("--discr_start_epoch", type=int, default=5,
                        help="Starting epoch for discriminator training.")
    parser.add_argument('--weight_decay', '--wd', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-5)')
    parser.add_argument("--scheduler_step_size", type=int, default=5,
                        help="step size for the both schedulers")
    parser.add_argument("--static_weight", type=float, default=5.,
                        help="static weight for calculating loss")
    parser.add_argument("--dynamic_weight", type=float, default=15.,
                        help="dynamic weight for calculating loss")
    parser.add_argument("--occ_map_size", type=int, default=256,
                        help="size of topview occupancy map")
    parser.add_argument("--cam_height", type=int, default=1,
                        help="camera height from ground in metres")
    parser.add_argument("--obstacle_height", type=int, default=1,
                        help="obstacle height threshold from ground")
    parser.add_argument("--num_class", type=int, default=2,
                        help="Number of classes")
    parser.add_argument("--num_epochs", type=int, default=120,
                        help="Max number of training epochs")
    parser.add_argument("--log_frequency", type=int, default=5,
                        help="Log files every x epochs")
    parser.add_argument("--train_workers", type=int, default=8,
                        help="Number of cpu workers for train dataloader")
    parser.add_argument("--val_workers", type=int, default=8,
                        help="Number of cpu workers for val dataloader")
    parser.add_argument("--osm_path", type=str, default="./data/osm",
                        help="OSM path")
    parser.add_argument('--log_root', type=str, default=os.getcwd() + '/log')
    parser.add_argument('--model_split_save', type=bool, default=True)

    configs = edict(vars(parser.parse_args()))

    return configs


def get_eval_args():
    parser = argparse.ArgumentParser(description="Evaluation options")
    parser.add_argument("--data_path", type=str, default="./data",
                        help="Path to the root data directory")
    parser.add_argument("--pretrained_path", type=str, default="./models/",
                        help="Path to the pretrained model")
    parser.add_argument("--osm_path", type=str, default="./data/osm",
                        help="OSM path")
    parser.add_argument(
        "--split",
        type=str,
        choices=[
            "argo",
            "3Dobject",
            "odometry",
            "raw",
            "gibson"],
        help="Data split for training/validation")
    parser.add_argument("--ext", type=str, default="png",
                        help="File extension of the images")
    parser.add_argument("--height", type=int, default=1024,
                        help="Image height")
    parser.add_argument("--width", type=int, default=1024,
                        help="Image width")
    parser.add_argument(
        "--type",
        type=str,
        choices=[
            "both",
            "static",
            "dynamic"],
        help="Type of model being trained")
    parser.add_argument("--occ_map_size", type=int, default=256,
                        help="size of topview occupancy map")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of cpu workers for dataloaders")
    parser.add_argument("--out_dir", type=str,
                        default="output")
    parser.add_argument("--model_name", type=str, default="crossView",
                        help="Model Name with specifications")
    parser.add_argument("--num_class", type=int, default=2,
                        help="Number of classes")
    configs = edict(vars(parser.parse_args()))

    return configs
