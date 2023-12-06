#!/bin/bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base_dir="${script_dir%/*}"

base_src_path="${base_dir}"

train_iters=3
log_interval=3

# Use the command "./run_model.sh --help" to display usage information
run_cmd="${script_dir}/run_model.sh \
        --model=125M \
        --seqlen=1024 \
        --gpus-per-node=8 \
        --batch-size=16 \
        --tp-size=4 \
        --pp-size=1 \
        --zero-stage=0 \
        --train-iters=${train_iters} \
        --base-src-path=${base_src_path} \
        --base-data-path=${base_dir}/dataset \
        --base-output-path=${base_dir}/output \
        --log-interval=${log_interval}"

echo "${run_cmd}"
eval "${run_cmd}"

# head_size > 128
#run_cmd="./pretrain_gpt_with_mp.sh 6B 2 2 1 1"
#echo "${run_cmd}"
#eval "${run_cmd}"

