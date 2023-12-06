#!/bin/bash

set -x

ds_commit=d24629f4

pushd .
[ -d ./DeepSpeed ] && rm -r ./DeepSpeed
git clone --recursive https://github.com/microsoft/DeepSpeed.git
cd ./DeepSpeed
git checkout ${ds_commit}
pip install .[dev,1bit,autotuning]
popd

pushd .
[ -d ./flash-attention ] && rm -r ./flash-attention
git clone --recursive https://github.com/ROCmSoftwarePlatform/flash-attention.git
cd ./flash-attn
patch /opt/conda/envs/py_3.8/lib/python3.8/site-packages/torch/utils/hipify/hipify_python.py hipify_patch.patch
python setup.py install
popd

set +x
