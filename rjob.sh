#!/bin/bash
export USER_NAME="lijiahang"
export JOB_DIR="/mnt/shared-storage-user/${USER_NAME}/jobs/mattersim/"
export NODES="node/gpu-l-lg-cmc-h-h200-0238.host.h.pjlab.org.cn"

num_nodes=1
name="mattersim"

chmod +x ${JOB_DIR}/run_parallel.sh
rjob submit \
--enable-sshd \
--name=${name} \
--gpu=8 \
--memory=128000 \
--cpu=64 \
--charged-group=omnimat_gpu \
--private-machine=group \
--mount=gpfs://gpfs1/${USER_NAME}:/mnt/shared-storage-user/${USER_NAME} \
--image=registry.h.pjlab.org.cn/ailab-omnimat/chenshuizhou-workspace:20250917184047 \
-P ${num_nodes} \
--host-network=true \
--preemptible=no \
-e DISTRIBUTED_JOB=true \
--custom-resources rdma/mlnx_shared=8 \
--positive-tags ${NODES} \
-- bash -exc "${JOB_DIR}/run.sh $1"