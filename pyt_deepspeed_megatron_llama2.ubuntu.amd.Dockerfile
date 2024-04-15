# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
ARG BASE_DOCKER=rocm/pytorch:latest
FROM $BASE_DOCKER
ENV WORKSPACE_DIR=/workspace
RUN mkdir -p $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR
ENV MAX_JOBS=128

##############################################################################
# Temporary Installation Directory
##############################################################################
ENV STAGE_DIR=/tmp
RUN mkdir -p ${STAGE_DIR}

##############################################################################
# Installation/Basic Utilities
##############################################################################
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common build-essential autotools-dev \
        nfs-common pdsh \
        cmake g++ gcc \
        curl wget vim tmux emacs less unzip \
        htop iftop iotop ca-certificates openssh-client openssh-server \
        rsync iputils-ping net-tools sudo

##############################################################################
# Installation Latest Git
##############################################################################
RUN add-apt-repository ppa:git-core/ppa -y && \
    apt-get update && \
    apt-get install -y git && \
    git --version

##############################################################################
# Client Liveness & Uncomment Port 22 for SSH Daemon
##############################################################################
# Keep SSH client alive from server side
# RUN echo "ClientAliveInterval 30" >> /etc/ssh/sshd_config
# RUN cp /etc/ssh/sshd_config ${STAGE_DIR}/sshd_config && \
#     sed "0,/^#Port 22/s//Port 22/" ${STAGE_DIR}/sshd_config > /etc/ssh/sshd_config

##############################################################################
# Python
##############################################################################
ENV DEBIAN_FRONTEND=noninteractive

# ##############################################################################
# # Some Packages
# ##############################################################################
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsndfile-dev \
        libjpeg-dev \
        libpng-dev \
        libaio-dev \
        screen \
        cargo \
        libopenmpi-dev \
        python3-dev \
        curl

# RUN apt update
# RUN apt install -y -V ca-certificates lsb-release wget
# RUN wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
# RUN apt install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
# RUN apt update
# RUN apt install -y -V libarrow-dev # For C++

RUN pip install psutil \
                yappi \
                cffi \
                ipdb \
                pandas \
                py3nvml \
                graphviz \
                astor \
                boto3 \
                tqdm \
                sentencepiece \
                msgpack \
                requests \
                pandas \
                sphinx \
                sphinx_rtd_theme \
                scipy \
                numpy \
                scikit-learn \
                ninja \
                datasets \
                nltk

RUN pip install matplotlib pyarrow --prefer-binary
# RUN apt install -y -V python3-matplotlib

# RUN pip install pyarrow \
#                 transformers \
#                 pandas \
#                 datasets \
#                 nltk \
#                 sentencepiece \
#                 pybind11 \
#                 einops \
#                 ninja

##############################################################################
# DeepSpeed
##############################################################################
RUN git clone https://github.com/microsoft/DeepSpeed.git ${STAGE_DIR}/DeepSpeed

# DeepSpeed with version v0.12.6 works with Megatron-DeepSpeed
WORKDIR ${STAGE_DIR}/DeepSpeed
RUN git checkout v0.12.6 && \ 
    pip install .[dev,1bit,autotuning]
RUN rm -rf ${STAGE_DIR}/DeepSpeed
RUN cd ~ && python3 -c "import deepspeed; print(deepspeed.__version__)"

##############################################################################
# Apex
##############################################################################
RUN git clone https://github.com/ROCm/apex.git ${STAGE_DIR}/apex 
WORKDIR ${STAGE_DIR}/apex
RUN python3 setup.py install --cpp_ext --cuda_ext
WORKDIR $WORKSPACE_DIR
RUN rm -rf ${STAGE_DIR}/apex

##############################################################################
# Megatron-DeepSpeed
##############################################################################
RUN git clone https://github.com/microsoft/Megatron-DeepSpeed.git ${STAGE_DIR}/Megatron-DeepSpeed

WORKDIR ${STAGE_DIR}/Megatron-DeepSpeed
RUN cd ${STAGE_DIR}/Megatron-DeepSpeed && \
       pip install -r megatron/core/requirements.txt

# ##############################################################################
# # Flash Attention
# ##############################################################################
RUN git clone https://github.com/ROCm/flash-attention.git ${STAGE_DIR}/flash-attention

WORKDIR ${STAGE_DIR}/flash-attention
RUN GPU_ARCHS=gfx90a python3 setup.py install
RUN rm -rf ${STAGE_DIR}/flash-attention

WORKDIR $WORKSPACE_DIR
RUN rm -rf ${STAGE_DIR}/DeepSpeed

ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

