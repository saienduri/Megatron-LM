#!/bin/bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base_dir="${script_dir%/*}"

export BASE_SRC_PATH="${base_dir}"
export BASE_DATA_PATH="${base_dir}/dataset"

[ -d ${BASE_DATA_PATH} ] || mkdir -p "${BASE_DATA_PATH}"

pushd .
cd ${BASE_DATA_PATH}
wget https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
xz -d oscar-1GB.jsonl.xz
popd

pip3 install nltk

python ${BASE_SRC_PATH}/tools/preprocess_data.py \
    --input ${BASE_DATA_PATH}/oscar-1GB.jsonl \
    --output-prefix ${BASE_DATA_PATH}/my-gpt2 \
    --vocab ${BASE_DATA_PATH}/gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file ${BASE_DATA_PATH}/gpt2-merges.txt \
    --append-eod \
    --workers 64
