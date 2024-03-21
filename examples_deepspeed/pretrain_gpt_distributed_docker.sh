#!/bin/bash

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=23456
export LOGLEVEL=INFO

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base_dir="${script_dir%/*}"
data_dir="${base_dir}/dataset"
output_dir="${base_dir}/output"

[ -d "${data_dir}" ] || mkdir -p "${data_dir}"
[ -d "${output_dir}" ] || mkdir -p "${output_dir}"

_config_env=(SLURM_JOB_NUM_NODES SLURM_NODEID SLURM_JOB_ID MASTER_ADDR MASTER_PORT)
mapfile -t _config_env < <(for v in "${_config_env[@]}"; do echo "--env=$v"; done)

# Build Docker image
# This helps set up the environment by installing DeepSpeed and the other dependencies.
cd "${base_dir}"

docker run ${_config_env[@]} --rm -tid --privileged --network=host --device=/dev/kfd --device=/dev/dri \
	--group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
	--ipc=host --shm-size 64G -v ~/:/dockerx -w /root \
	--name=megatron-deepspeed corescientificai/megatron:megatron_mi300x_mNode_6.2.0-13595

docker exec megatron-deepspeed /root/Megatron-Deepspeed-PoC/run.sh
docker stop megatron-deepspeed
