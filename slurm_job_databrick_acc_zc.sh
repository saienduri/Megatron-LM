#!/bin/bash

#NODES=

#SBATCH --job-name=zc_databrick
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gres=gpu:8
#SBATCH --time=00-10:00:00             #specify time for the job
#SBATCH --partition=amd-aig
#SBATCH --account=amd-aig
#SBATCH --nodelist=useocpm2m-401-004,useocpm2m-401-012
#######SBATCH --nodelist=useocpm2m-401-003,useocpm2m-401-004,useocpm2m-401-012,useocpm2m-401-[008-009],useocpm2m-401-011,useocpm2m-401-[013-016] #specify specific nodes (ML Pref) + 004+012

######SBATCH --nodelist=useocpm2m-401-008,useocpm2m-401-009
######SBATCH --nodelist=useocpm2m-401-003,useocpm2m-401-004,useocpm2m-401-012,useocpm2m-401-[008-009],useocpm2m-401-011,useocpm2m-401-[013-016] #specify specific nodes (ML Pref) + 004+012


#######SBATCH --nodelist=useocpm2m-401-003,useocpm2m-401-[008-009],useocpm2m-401-011,useocpm2m-401-[013-016] #specify specific nodes (ML Pref)

######SBATCH --nodelist=useocpm2m-401-008,useocpm2m-401-009
######SBATCH --nodelist=useocpm2m-401-004,useocpm2m-401-012
#####SBATCH --nodelist=useocpm2m-401-004,useocpm2m-401-012
#008, 009

######SBATCH --nodelist=useocpm2m-401-003,useocpm2m-401-[008-011],useocpm2m-401-[013-020],useocpm2m-401-[022-032] #specify specific nodes (LLM pre-training + MLPref)
#####SBATCH --nodelist=useocpm2m-401-[013-014] #specify nodes
##### 8, 9 ---> fail (can get:  nan     275.74285714285713      nan     nan)
##### 15, 16 ---> fail
##### 13, 14 ---> ?

##########SBATCH --nodelist=useocpm2m-401-003,useocpm2m-401-[008-009],useocpm2m-401-011,useocpm2m-401-[013-016] #specify specific nodes (ML Pref)
#########SBATCH --nodelist=useocpm2m-401-003,useocpm2m-401-[008-011],useocpm2m-401-[013-020],useocpm2m-401-[022-032] #specify specific nodes (LLM pre-training + MLPref)




head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
######
###
master_port=$((20000 + $RANDOM % 40000))
###
#master_port=51310
###
# Function to find an available port
#find_available_port() {
#    local port
#    while true; do
#        port=$((20000 + RANDOM % 40000))
#        (echo >/dev/tcp/localhost/$port) >/dev/null 2>&1 || { echo $port; return 0; }
#    done
#}
#master_port=$(find_available_port)
###
######
#echo "Master Address: $master_addr"

head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Node IP: $head_node_ip"

export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_DEBUG=INFO
export NCCL_ENABLE_DMABUF_SUPPORT=1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9

#export LD_LIBRARY_PATH=/path/to/lib:$LD_LIBRARY_PATH

# train_acc_loss_databrick.sh v.s train_acc_loss_databrick_test.sh (modified as Jiang push on the github)


# /mnt/m2m_nobackup is the local storage on the nodes
# srun -l apptainer exec --bind /mnt/m2m_nobackup:/mnt/m2m_nobackup --rocm olmo.sif python test.py

# APPTAINER_CACHEDIR=/mnt/m2m_nobackup/temporary-cache-${USER}
# srun -l apptainer exec --bind $APPTAINER_CACHEDIR:$APPTAINER_CACHEDIR --rocm apptainer_images/olmo_convert.sif python test.py
# srun -l apptainer exec --bind /mnt/m2m_nobackup:/mnt/m2m_nobackup --rocm ../apptainer_images/olmo_convert.sif python test.py
#apptainer exec --bind /mnt/m2m_nobackup/yushengsu:/mnt/m2m_nobackup/yushengsu:rw,$HOME:$HOME:rw $HOME/apptainer_built_images/rocm_pytorch_private_exec_dash_pretuned_nightly_inai_FA_ck_v0_1_1_TE.sif bash run_databrick.sh


#[1]
#!!!!!!!!!!!!!!!!!!!!!
#srun -l apptainer exec --bind /mnt/m2m_nobackup/yushengsu:/mnt/m2m_nobackup/yushengsu:rw,$HOME:$HOME:rw $HOME/apptainer_built_images/rocm_pytorch_private_exec_dash_pretuned_nightly_inai_FA_ck_v0_1_1_TE.sif bash train_acc_loss_databrick_acc.sh MBS=5 BS=80 TP=1 PP=1 MODEL_SIZE=8 SEQ_LENGTH=2048 NO_TORCH_COMPILE=1 MASTER_ADDR=$head_node_ip NNODES=$SLURM_NNODES MASTER_PORT=$master_port TOTAL_ITERS=10

#TP=8, PP=1 --> will have NCCL problem (use this to debug)

#[1]
# Refer to here: https://amdcloud.sharepoint.com/:x:/r/sites/AIG/AIMA/_layouts/15/Doc.aspx?sourcedoc=%7B971FAA60-B8DF-4860-A7EA-955538638D41%7D&file=Databricks%20training%20project.xlsx&wdOrigin=TEAMS-MAGLEV.p2p_ns.rwc&action=default&mobileredirect=true
gobal_batch_size=80
micro_batch_size=5
model_size=8
tp=1 #1(), 2(), 4(), 8 (o)
pp=1
no_torch_compile=0 # (in acc exp is 1), 0 will be faster but acc will ...?

#TP=8 PP=1 (TP>8 will have the bug) --> NCCL problem --> now: TP=1, PP=1

#fp16 --> #TE_FP16=0
# Test (train_acc_loss_databrick_acc.sh)
#srun -l apptainer exec --bind /mnt/m2m_nobackup/yushengsu:/mnt/m2m_nobackup/yushengsu:rw,$HOME:$HOME:rw $HOME/apptainer_built_images/rocm_pytorch_private_exec_dash_pretuned_nightly_inai_FA_ck_v0_1_1_TE.sif bash train_acc_loss_databrick.sh MBS=$micro_batch_size BS=$gobal_batch_size TP=$tp PP=$pp MODEL_SIZE=$model_size SEQ_LENGTH=2048 NO_TORCH_COMPILE=$no_torch_compile MASTER_ADDR=$head_node_ip NNODES=$SLURM_NNODES MASTER_PORT=$master_port TOTAL_ITERS=20 TE_FP16=0


#sft
# tune_pt (tune_basetrain_databrick.sh)
#srun -l apptainer exec --bind /mnt/m2m_nobackup/yushengsu:/mnt/m2m_nobackup/yushengsu:rw,$HOME:$HOME:rw $HOME/apptainer_built_images/rocm_pytorch_private_exec_dash_pretuned_nightly_inai_FA_ck_v0_1_1_TE.sif bash tune_basetrain_databrick.sh MBS=$micro_batch_size BS=$gobal_batch_size TP=$tp PP=$pp MODEL_SIZE=$model_size SEQ_LENGTH=2048 NO_TORCH_COMPILE=$no_torch_compile MASTER_ADDR=$head_node_ip NNODES=$SLURM_NNODES MASTER_PORT=$master_port TOTAL_ITERS=20 TE_FP16=0



#fp8 --> #TE_FP16=1
# Test (train_acc_loss_databrick_acc.sh)
#srun -l apptainer exec --bind /mnt/m2m_nobackup/yushengsu:/mnt/m2m_nobackup/yushengsu:rw,$HOME:$HOME:rw $HOME/apptainer_built_images/rocm_pytorch-private:exec_dash_pretuned_nightly_inai_FA_ck_v0.1.1_TE_with_CP.sif bash train_acc_loss_databrick_acc.sh MBS=$micro_batch_size BS=$gobal_batch_size TP=$tp PP=$pp MODEL_SIZE=$model_size SEQ_LENGTH=2048 NO_TORCH_COMPILE=$no_torch_compile MASTER_ADDR=$head_node_ip NNODES=$SLURM_NNODES MASTER_PORT=$master_port TOTAL_ITERS=20 TE_FP16=1

# tune_pt (tune_basetrain_databrick.sh)
srun -l apptainer exec --bind /mnt/m2m_nobackup/yushengsu:/mnt/m2m_nobackup/yushengsu:rw,$HOME:$HOME:rw $HOME/apptainer_built_images/rocm_pytorch-private:exec_dash_pretuned_nightly_inai_FA_ck_v0.1.1_TE_with_CP.sif bash tune_basetrain_databrick.sh MBS=$micro_batch_size BS=$gobal_batch_size TP=$tp PP=$pp MODEL_SIZE=$model_size SEQ_LENGTH=2048 NO_TORCH_COMPILE=$no_torch_compile MASTER_ADDR=$head_node_ip NNODES=$SLURM_NNODES MASTER_PORT=$master_port TOTAL_ITERS=20 TE_FP16=1





# FP-8 docker: rocm/pytorch-private:exec_dash_pretuned_nightly_inai_FA_ck_v0.1.1_TE_with_CP
# ADD: TE_FP16=1
# run this one: train70b_acc_loss_databrick.sh
# NO_TORCH_COMPILE=1
# llam3 70B --> 2B
# TOTAL_ITERS=5,000; 10,000
# TE_FP16 --> mean FP8-GEMM
# with TP=8 PP=1
#2k sequence length



#BS:80, num_GPU:8, MBS: 2 --> 5 --> 10 [node=1] (rest times is accumulation)
#BS:56, num_GPU:8, MBS:7 [node=1]

#BS:160, num_GPU:16, MBS:2 --> 5 --> 10 [node=2] (rest times is accumulation)
#BS:112, num_GPU:16, MBS:7 [node=2]

#BS:320, num_GPU:32, MBS:2 --> 5 --> 10 [node=4] (rest time is accumulation)
#BS:224, num_GPU:32, MBS:7 [node=4]

#BS:640, num_GPU:64, MBS:2 --> 5 --> 10 [node=8] (rest time is accumulation)
#BS:448, num_GPU:64, MBS:7 [node=8]


# MBS: micro_batch_size ==> BS % (num_GPUs * MBS / TP / PP) --should be-->  (int)
# global batch size (80) is not divisible by micro batch size (5) times data parallel size (32)

#Global Batch Size = Micro Batch Size × Number of Pipeline Stages × Number of Data Parallel Replicas.

#mv out.csv out_${NODES}.csv
#mv out.csv out_4.csv
