#!/bin/bash


#SBATCH --job-name=databrick
#SBATCH --nodes=2
#SBATCH --cpus-per-gpu=16
#SBATCH --gres=gpu:8
#SBATCH --time=00-10:00:00             #specify time for the job
#SBATCH --partition=amd-aig
#SBATCH --account=amd-aig
#SBATCH --nodelist=useocpm2m-401-003,useocpm2m-401-[008-009],useocpm2m-401-011,useocpm2m-401-[013-016]    #specify specific nodes, if you want those specific nodes


head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#echo "Master Address: $master_addr"

head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Node IP: $head_node_ip"

export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_DEBUG=INFO
export NCCL_ENABLE_DMABUF_SUPPORT=1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9


# /mnt/m2m_nobackup is the local storage on the nodes
# srun -l apptainer exec --bind /mnt/m2m_nobackup:/mnt/m2m_nobackup --rocm olmo.sif python test.py

# APPTAINER_CACHEDIR=/mnt/m2m_nobackup/temporary-cache-${USER}
# srun -l apptainer exec --bind $APPTAINER_CACHEDIR:$APPTAINER_CACHEDIR --rocm apptainer_images/olmo_convert.sif python test.py
# srun -l apptainer exec --bind /mnt/m2m_nobackup:/mnt/m2m_nobackup --rocm ../apptainer_images/olmo_convert.sif python test.py
#apptainer exec --bind /mnt/m2m_nobackup/yushengsu:/mnt/m2m_nobackup/yushengsu:rw,$HOME:$HOME:rw $HOME/apptainer_built_images/rocm_pytorch_private_exec_dash_pretuned_nightly_inai_FA_ck_v0_1_1_TE.sif bash run_databrick.sh


# NODE_RANK=0 NODE_RANK=1 NODE_RANK=2 NODE_RANK=3
# nodes=1, nodes=2, nodes=3, nodes=4
srun -l apptainer exec --bind /mnt/m2m_nobackup/yushengsu:/mnt/m2m_nobackup/yushengsu:rw,$HOME:$HOME:rw $HOME/apptainer_built_images/rocm_pytorch_private_exec_dash_pretuned_nightly_inai_FA_ck_v0_1_1_TE.sif bash tune_basetrain_databrick.sh MBS=5 BS=80 TP=1 PP=1 MODEL_SIZE=8 NO_TORCH_COMPILE=0 MASTER_ADDR=$head_node_ip NNODES=$SLURM_NNODES 

