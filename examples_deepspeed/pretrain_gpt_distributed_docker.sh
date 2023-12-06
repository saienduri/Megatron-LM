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
podman build -f Dockerfile -t megatron/deepspeed:rocm .

# If there exists an obsolete container, remove it.
if [ "$(podman ps -aq -f name=megatron-deepspeed)" ]; then
    podman rm megatron-deepspeed
fi

podman run ${_config_env[@]} -tid --privileged --network=host --shm-size=64GB \
        -v /shareddata:/shareddata \
        -v ${data_dir}:/dataset \
        -v ${output_dir}:/output \
        --name=megatron-deepspeed megatron/deepspeed:rocm

podman exec megatron-deepspeed /megatron-deepspeed/scripts/run_megatron_example.sh
