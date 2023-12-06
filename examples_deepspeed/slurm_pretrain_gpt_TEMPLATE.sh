#!/bin/bash
#SBATCH --job-name=megatron
#SBATCH -o %x-%j_%N.out
#SBATCH -e %x-%j_%N.err
#SBATCH --partition=PARTITION_NAME
#SBATCH --nodes=NUM_NODES
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --gpus-per-node=8            # number of cores per tasks
#SBATCH --time=23:59:59              # maximum execution time (HH:MM:SS)
#SBATCH --exclusive

srun -N NUM_NODES $@
