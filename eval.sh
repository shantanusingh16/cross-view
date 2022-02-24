CUDA_VISIBLE_DEVICES=2 python3 eval.py --data_path /scratch/shantanu/gibson4/new --split gibson4 \
    --height 512 --width 512 --type static --occ_map_size 128 --num_class 3 \
    --model_name transformer_discr \
    --pretrained_path /scratch/jaidev/basic_discr/transformer_discr/weights_63/ \
    --out_dir /scratch/jaidev/outputs/basic_discr/ \
    --bev_dir /scratch/shantanu/gibson4/dilated_partialmaps/
