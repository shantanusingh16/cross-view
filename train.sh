CUDA_VISIBLE_DEVICES=2 python3 train.py --model_name transformer_discr --data_path \
/scratch/shantanu/gibson4/new --split gibson4 --width 512 --height 512 --num_class 3 --type static \
--static_weight 1 --occ_map_size 128 --log_frequency 1 --log_root /scratch/jaidev/basic_discr \
--save_path /scratch/jaidev/basic_discr --semantics_dir None --chandrakar_input_dir None \
--floor_path None --batch_size 8 --num_epochs 100 --lr_steps 50 --lr 1e-4 --lr_transform 1e-3 \
--load_weights_folder None --bev_dir /scratch/shantanu/gibson4/dilated_partialmaps \
--train_workers 15 --val_workers 8
