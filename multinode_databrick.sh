#!/bin/bash

# Set necessary environment variables
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_DEBUG=INFO
export NCCL_ENABLE_DMABUF_SUPPORT=1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9

# Define the number of nodes and GPUs
NUM_NODES=2
GPUS_PER_NODE=8
MASTER_PORT=$((20000 + $RANDOM % 40000))

# Get the IP address of the head node
head_node=$(hostname --ip-address)
echo "Node IP: $head_node"

# Execute the training script using torchrun
torchrun --nnodes=$NUM_NODES --nproc_per_node=$GPUS_PER_NODE --rdzv_id=1234 --rdzv_backend=c10d --rdzv_endpoint=$head_node:$MASTER_PORT \
    --module apptainer exec --bind /mnt/m2m_nobackup/yushengsu:/mnt/m2m_nobackup/yushengsu:rw,$HOME:$HOME:rw $HOME/apptainer_built_images/rocm_pytorch_private_exec_dash_pretuned_nightly_inai_FA_ck_v0_1_1_TE.sif \
    tune_basetrain_databrick.sh MBS=5 BS=160 TP=1 PP=1 MODEL_SIZE=8 SEQ_LENGTH=2048 NO_TORCH_COMPILE=0 MASTER_ADDR=$head_node NNODES=$NUM_NODES MASTER_PORT=$MASTER_PORT TOTAL_ITERS=20

