#!/bin/bash

curr_task="WIKITEXT103"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base_dir="${script_dir%/*}"
data_dir="${base_dir}/dataset"
log_dir="${base_dir}/log"

[ -d "${log_dir}" ] || mkdir -p "${log_dir}"
log_file="${log_dir}/eval_wikitext_gpt2_345m_seqlen_${seqlen}_mbs_${micro_batch_size}_tp_${tp_size}_expert_${experts}_node_${nodes}_gpu_${gpus}.log"

vocab_file="${data_dir}/gpt2-vocab.json"
merge_file="${data_dir}/gpt2-merges.txt"
valid_data="${data_dir}/wikitext-103/wiki.test.tokens"
checkpoint_path="${base_dir}/checkpoints/gpt2_345m"

micro_batch_size=8
num_layers=24
hidden_size=1024
num_attn_heads=16
seqlen=1024
log_interval=10

num_nodes=1
num_gpus=8

launch_cmd="deepspeed --num_nodes $num_nodes --num_gpus $num_gpus"

program_cmd="${base_dir}/tasks/main.py \
       --task $curr_task \
       --tokenizer-type GPT2BPETokenizer \
       --strict-lambada \
       --vocab-file $vocab_file \
       --merge-file $merge_file \
       --valid-data $valid_data \
       --load $checkpoint_path \
       --micro-batch-size $micro_batch_size \
       --num-layers $num_layers \
       --hidden-size $hidden_size \
       --num-attention-heads $num_attn_heads \
       --seq-length $seqlen \
       --max-position-embeddings $seqlen \
       --log-interval $log_interval \
       --fp16 \
       --no-load-optim \
       --no-load-rng"

echo "$launch_cmd $program_cmd"      | tee "${log_file}"
eval "$launch_cmd $program_cmd" 2>&1 | tee -a "${log_file}"
