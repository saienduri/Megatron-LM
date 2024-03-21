#!/bin/bash

set -ex

partition_name=${1:-"1CN224C8G1H_MI300X_Ubuntu22"}
num_nodes=${2:-1}

pattern='^[0-9]+$'
if ! [[ ${num_nodes} =~ ${pattern} ]]; then
    echo "Error: num_nodes (${num_nodes}) not a number"
    exit
elif [ ${num_nodes} -lt 1 ]; then
    echo "Error: num_nodes must be larger than or equal to 1"
    exit
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
template_slurm="${script_dir}/slurm_pretrain_gpt_TEMPLATE.sh"
if [ ${num_nodes} -eq 1 ]; then
    slurm_script="${script_dir}/slurm_pretrain_gpt_single_node.sh"
else
    slurm_script="${script_dir}/slurm_pretrain_gpt_${num_nodes}nodes.sh"
fi

sed "s/PARTITION_NAME/${partition_name}/" ${template_slurm} \
        | sed "s/NUM_NODES/${num_nodes}/" \
	    > ${slurm_script}

run_cmd="sbatch ${slurm_script} pretrain_gpt_distributed_docker.sh"
echo "${run_cmd}"
eval "${run_cmd}"

rm -f "${slurm_script}"
