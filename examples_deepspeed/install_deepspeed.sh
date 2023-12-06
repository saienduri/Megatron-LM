#!/bin/bash

ds_commit=d24629f4
count=$(pip3 list | grep -c "deepspeed")

if [ ${count} -eq 0 ]; then
    echo "Installing DeepSpeed..."
    pushd .
    [ -d /tmp/DeepSpeed  ] && rm -rf /tmp/DeepSpeed
    git clone --recursive https://github.com/microsoft/DeepSpeed.git /tmp/DeepSpeed
    cd /tmp/DeepSpeed
    git checkout ${ds_commit}
    pip install .[dev,1bit,autotuning]
    popd
fi
