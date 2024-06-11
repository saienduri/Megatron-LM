#!/bin/bash

## TODO: Follow README to prepare your dataset

## Prepare the tuning code
# git clone https://github.com/ROCm/pytorch_afo_testkit.git
# cd pytorch_afo_testkit && pip install -e . && cd ..

## Run forward, gather information and generate GEMM tuning result
ROCBLAS_LAYER=4 bash train_gpt3_1step.sh 2>&1 | grep "\- { rocblas_function:" | uniq | tee rocblas.yaml
python pytorch_afo_testkit/afo/tools/tuning/tune_from_rocblasbench.py rocblas.yaml --cuda_device 0 1 2 3 4 5 6 7
rm -rf outputs/gpt3_13b

## Training with GEMM tuning
export PYTORCH_TUNABLEOP_FILENAME=full_tuned%d.csv
export PYTORCH_TUNABLEOP_TUNING=0
export PYTORCH_TUNABLEOP_ENABLED=1
bash train_gpt3.sh
