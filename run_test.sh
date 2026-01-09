#!/usr/bin/env bash
set -ex

train_data_path="/mnt/shared-storage-gpfs2/ailab-omnimat-shared/lijielan/datasets/mptrj/oxides/train/oxide_train.pkl"
valid_data_path="/mnt/shared-storage-gpfs2/ailab-omnimat-shared/lijielan/datasets/mptrj/oxides/valid/oxide_valid.pkl"
# torchrun \
#     --nnodes=1 \
#     --nproc_per_node=1 \
#     --node_rank=0 \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint="127.0.0.1:29500" \
#     --rdzv_id="mattersim" \

python train_m3gnet.py \
    --train_data_path ${train_data_path} \
    --valid_data_path ${valid_data_path} \
    --save_checkpoint \
    --device "cuda" \
    --cutoff 5.0 \
    --threebody_cutoff 4.0 \
    --epochs 2000 \
    --batch_size 256 \
    --lr 8e-4 \
    --step_size 50 \
    --include_forces \
    --include_stresses \
    --force_loss_ratio 1.0 \
    --stress_loss_ratio 0.1 \
    --early_stop_patience 50 \
    --seed 42 \
    --normalize \
    --scale_key "per_species_forces_rms" \
    --shift_key "per_species_energy_mean_linear_reg" \
    --wandb \
    --wandb_project "m3gnet_test" \
    --wandb_dir "/mnt/shared-storage-user/lijiahang/wandb/mattersim"