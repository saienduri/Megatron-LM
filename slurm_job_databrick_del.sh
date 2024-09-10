#!/bin/bash

#NODES=

#SBATCH --job-name=test_databrick
#SBATCH --nodes=6
#SBATCH --cpus-per-gpu=16
#SBATCH --gres=gpu:8
#SBATCH --time=00-10:00:00             #specify time for the job
#SBATCH --partition=amd-aig
#SBATCH --account=amd-aig
#SBATCH --nodelist=useocpm2m-401-[008-009],useocpm2m-401-011,useocpm2m-401-[013-016] #specify nodes

##########SBATCH --nodelist=useocpm2m-401-003,useocpm2m-401-[008-009],useocpm2m-401-011,useocpm2m-401-[013-016] #specify specific nodes (ML Pref)

###########SBATCH --nodelist=useocpm2m-401-003,useocpm2m-401-[008-011],useocpm2m-401-[013-020],useocpm2m-401-[022-032] #specify specific nodes, if you want those specific nodes

##############SBATCH --nodelist=useocpm2m-401-[008-010],useocpm2m-401-[013-014],useocpm2m-401-[017-020],useocpm2m-401-[022-032] #specify specific nodes, if you want those specific nodes

head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
master_port=$((20000 + $RANDOM % 40000))
#echo "Master Address: $master_addr"

head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Node IP: $head_node_ip"

export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_DEBUG=INFO
export NCCL_ENABLE_DMABUF_SUPPORT=1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9

# train70b_acc_loss_databrick.sh v.s train70b_acc_loss_databrick_test.sh (modified as Jiang push on the github)


# /mnt/m2m_nobackup is the local storage on the nodes
# srun -l apptainer exec --bind /mnt/m2m_nobackup:/mnt/m2m_nobackup --rocm olmo.sif python test.py

# APPTAINER_CACHEDIR=/mnt/m2m_nobackup/temporary-cache-${USER}
# srun -l apptainer exec --bind $APPTAINER_CACHEDIR:$APPTAINER_CACHEDIR --rocm apptainer_images/olmo_convert.sif python test.py
# srun -l apptainer exec --bind /mnt/m2m_nobackup:/mnt/m2m_nobackup --rocm ../apptainer_images/olmo_convert.sif python test.py
#apptainer exec --bind /mnt/m2m_nobackup/yushengsu:/mnt/m2m_nobackup/yushengsu:rw,$HOME:$HOME:rw $HOME/apptainer_built_images/rocm_pytorch_private_exec_dash_pretuned_nightly_inai_FA_ck_v0_1_1_TE.sif bash run_databrick.sh


#[1]
#!!!!!!!!!!!!!!!!!!!!!
#srun -l apptainer exec --bind /mnt/m2m_nobackup/yushengsu:/mnt/m2m_nobackup/yushengsu:rw,$HOME:$HOME:rw $HOME/apptainer_built_images/rocm_pytorch_private_exec_dash_pretuned_nightly_inai_FA_ck_v0_1_1_TE.sif bash train70b_acc_loss_databrick_test.sh MBS=5 BS=80 TP=1 PP=1 MODEL_SIZE=8 SEQ_LENGTH=2048 NO_TORCH_COMPILE=1 MASTER_ADDR=$head_node_ip NNODES=$SLURM_NNODES MASTER_PORT=$master_port TOTAL_ITERS=10


#[2]
#!!!!!!!!!!!!!!!!!!!!
#srun -l apptainer exec --bind /mnt/m2m_nobackup/yushengsu:/mnt/m2m_nobackup/yushengsu:rw,$HOME:$HOME:rw $HOME/apptainer_built_images/rocm_pytorch_private_exec_dash_pretuned_nightly_inai_FA_ck_v0_1_1_TE.sif bash train70b_acc_loss_databrick_test.sh MBS=5 BS=160 TP=1 PP=1 MODEL_SIZE=8 SEQ_LENGTH=2048 NO_TORCH_COMPILE=1 MASTER_ADDR=$head_node_ip NNODES=$SLURM_NNODES MASTER_PORT=$master_port TOTAL_ITERS=10


#[4]
#srun -l apptainer exec --bind /mnt/m2m_nobackup/yushengsu:/mnt/m2m_nobackup/yushengsu:rw,$HOME:$HOME:rw $HOME/apptainer_built_images/rocm_pytorch_private_exec_dash_pretuned_nightly_inai_FA_ck_v0_1_1_TE.sif bash train70b_acc_loss_databrick_test.sh MBS=5 BS=320 TP=1 PP=1 MODEL_SIZE=8 SEQ_LENGTH=2048 NO_TORCH_COMPILE=1 MASTER_ADDR=$head_node_ip NNODES=$SLURM_NNODES MASTER_PORT=$master_port TOTAL_ITERS=10


#[6]
srun -l apptainer exec --bind /mnt/m2m_nobackup/yushengsu:/mnt/m2m_nobackup/yushengsu:rw,$HOME:$HOME:rw $HOME/apptainer_built_images/rocm_pytorch_private_exec_dash_pretuned_nightly_inai_FA_ck_v0_1_1_TE.sif bash train70b_acc_loss_databrick_test.sh MBS=5 BS=480 TP=1 PP=1 MODEL_SIZE=8 SEQ_LENGTH=2048 NO_TORCH_COMPILE=0 MASTER_ADDR=$head_node_ip NNODES=$SLURM_NNODES MASTER_PORT=$master_port TOTAL_ITERS=100000

#[8]
#srun -l apptainer exec --bind /mnt/m2m_nobackup/yushengsu:/mnt/m2m_nobackup/yushengsu:rw,$HOME:$HOME:rw $HOME/apptainer_built_images/rocm_pytorch_private_exec_dash_pretuned_nightly_inai_FA_ck_v0_1_1_TE.sif bash train70b_acc_loss_databrick_test.sh MBS=5 BS=640 TP=1 PP=1 MODEL_SIZE=8 SEQ_LENGTH=2048 NO_TORCH_COMPILE=1 MASTER_ADDR=$head_node_ip NNODES=$SLURM_NNODES MASTER_PORT=$master_port TOTAL_ITERS=1000000


#BS:80, num_GPU:8, MBS: 2 --> 5 --> 10 [node=1] (rest times is accumulation)
#BS:56, num_GPU:8, MBS:7 [node=1]

#BS:160, num_GPU:16, MBS:2 --> 5 --> 10 [node=2] (rest times is accumulation)
#BS:112, num_GPU:16, MBS:7 [node=2]

#BS:320, num_GPU:32, MBS:2 --> 5 --> 10 [node=4] (rest time is accumulation)
#BS:224, num_GPU:32, MBS:7 [node=4]

#BS:640, num_GPU:64, MBS:2 --> 5 --> 10 [node=8] (rest time is accumulation)
#BS:448, num_GPU:64, MBS:7 [node=8]


# MBS: micro_sbastch_size ==> BS % (num_GPUs * MBS / TP / PP) --should be-->  (int)
# global batch size (80) is not divisible by micro batch size (5) times data parallel size (32)

#Global Batch Size = Micro Batch Size × Number of Pipeline Stages × Number of Data Parallel Replicas.

#mv out.csv out_${NODES}.csv
#mv out.csv out_4.csv
