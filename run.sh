#!/usr/bin/env bash
set -ex

# --------------- User specifications ---------------
export USER_NAME="lijiahang"
export ENV_NAME="mattersim"
export PROJECT_NAME="mattersim"
stage=$1  # lf or hf, take the first argument to the script

# --------------- Paths & Conda ---------------
export JOB_DIR="/mnt/shared-storage-user/${USER_NAME}/jobs/mattersim/"
cd ${JOB_DIR}
export PATH="/mnt/shared-storage-user/${USER_NAME}/miniconda3/bin:$PATH"
. /mnt/shared-storage-user/${USER_NAME}/miniconda3/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# --------------- W&B: offline to local disk ---------------
export WANDB_MODE=offline
export WANDB_PROJECT=${PROJECT_NAME}
export WANDB_DIR="/mnt/shared-storage-user/${USER_NAME}/wandb/${JOB_ID:-local_job}/node${NODE_RANK:-0}"
export WANDB_RUN_GROUP="${JOB_ID:-mp_group}"

NNODES="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${PROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"

# --------------- model argument setting ---------------

####################
# jiahang: NOTE
# The training script comes from /home/lijiahang/mattersim/unary/m3gnet-omat24-rjob-lr8e-4-stepsize50/run.sh
# This experiment shows best validation loss among others and acceptable training time (170 min)
# be noted that in the original experiment the number of gpu is 2, meaning that the effective bs is 512 if the bs is set to 256.
# we will use 8 gpu to train, leading to much large effective bs = 256 * 8.
####################

if [ ${stage} == "lf" ]; then
    train_data_path="/mnt/shared-storage-gpfs2/ailab-omnimat-shared/lijielan/datasets/mptrj/oxides/train/oxide_train.pkl"
    valid_data_path="/mnt/shared-storage-gpfs2/ailab-omnimat-shared/lijielan/datasets/mptrj/oxides/valid/oxide_valid.pkl"
elif [ ${stage} == "hf" ]; then
    train_data_path="/mnt/shared-storage-gpfs2/ailab-omnimat-shared/lijielan/datasets/mptrj/oxides_r2scan/train/oxide_r2scan_train.pkl"
    valid_data_path="/mnt/shared-storage-gpfs2/ailab-omnimat-shared/lijielan/datasets/mptrj/oxides_r2scan/valid/oxide_r2scan_valid.pkl"
fi

torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --node_rank="$NODE_RANK" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_id="${JOB_ID:-mattersim}" \
    train_m3gnet.py \
    --distributed \
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
    --wandb_project "m3gnet_run" \
    --wandb_dir "/mnt/shared-storage-user/lijiahang/wandb/mattersim"