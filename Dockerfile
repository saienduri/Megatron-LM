ARG FROM_IMAGE_NAME=docker.io/rocm/pytorch:rocm5.6_ubuntu20.04_py3.8_pytorch_2.0.1
ARG ds_commit=d24629f4

FROM ${FROM_IMAGE_NAME} as base-image

ENV PYTORCH_ROCM_ARCH=gfx90a
RUN pip3 --no-cache-dir install nltk pybind11

COPY . /megatron-deepspeed
WORKDIR /megatron-deepspeed

# DeepSpped
RUN git clone --recursive https://github.com/microsoft/DeepSpeed.git /tmp/DeepSpeed && \
    cd /tmp/DeepSpeed && \
    git checkout ${ds_commit} && \
    pip install .[dev,1bit,autotuning]

# FlashAttention
RUN git clone --recursive https://github.com/ROCmSoftwarePlatform/flash-attention.git /tmp/flash-attention && \
    cd /tmp/flash-attention && \
    patch /opt/conda/envs/py_3.8/lib/python3.8/site-packages/torch/utils/hipify/hipify_python.py hipify_patch.patch && \
    python3 setup.py install

