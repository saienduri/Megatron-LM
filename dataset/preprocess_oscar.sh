#!/bin/bash

script_dir="$(cd "$(dirname "${BASE_SOURCE[0]}")" && pwd)"
main_dir="${script_dir%/*}"
data_dir="${main_dir}/dataset"

run_cmd="python3 \
    ${main_dir}/tools/preprocess_data.py \
    --input ${data_dir}/oscar-1GB.jsonl \
    --output-prefix ${data_dir}/oscar-gpt2 \
    --vocab ${data_dir}/gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file ${data_dir}/gpt2-merges.txt \
    --append-eod \
    --workers 8"

echo "${run_cmd}"
eval "${run_cmd}"

