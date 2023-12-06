#!/bin/bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base_dir="${script_dir%/*}"
data_dir="${base_dir}/dataset"
log_dir="${base_dir}/log"

vocab_file="${data_dir}/gpt2-vocab.json"
merge_file="${data_dir}/gpt2-merges.txt"
checkpoint_path="${base_dir}/checkpoints/gpt2_345m"

seqlen=1024
micro_batch_size=8
tp_size=1
experts=1
nodes=1
gpus=1

[ -d "${log_dir}" ] || mkdir -p "${log_dir}"
log_file="${log_dir}/gen_text_gpt2_345m_seqlen_${seqlen}_mbs_${micro_batch_size}_tp_${tp_size}_expert_${experts}_node_${nodes}_gpu_${gpus}.log"

#use_tutel=""
use_tutel="--use-tutel"

#ds_inference=""
#ds_inference="--ds-inference"

launch_cmd="deepspeed --num_nodes $nodes --num_gpus $gpus"
layers=24
hidden_size=1024
attn_heads=16
#experts1=${experts[$k]}

program_cmd="tools/generate_samples_gpt.py \
       --tensor-model-parallel-size $tp_size \
       --num-layers $layers \
       --hidden-size $hidden_size \
       --num-attention-heads $attn_heads \
       --max-position-embeddings $seqlen \
       --tokenizer-type GPT2BPETokenizer \
       --fp16 \
       --num-experts $experts \
       --mlp-type standard \
       --micro-batch-size $micro_batch_size \
       --seq-length $seqlen \
       --out-seq-length $seqlen \
       --temperature 1.0 \
       --vocab-file $vocab_file \
       --merge-file $merge_file \
       --genfile unconditional_samples.json \
       --top_p 0.9 \
       --log-interval 1 \
       --num-samples 0 \
       --load $checkpoint_path \
       $use_tutel $ds_inference"

echo "$launch_cmd $program_cmd"
eval "$launch_cmd $program_cmd" 2>&1 | tee ${log_file}
