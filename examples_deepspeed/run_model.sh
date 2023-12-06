#! /bin/bash

set -ex

# Function to display usage information
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "Main configs"
    echo "  --model=<name>             Model size (Default: 6.7B)"
    echo "  --seqlen=<value>           Sequence length (Default: 1024)"
    echo
    echo "Parallelism and optimization configs"
    echo "  --gpus-per-node=<value>    Number of GPUs per node (Default: 8)"
    echo "  --batch-size=<size>        Micro batch size (Default: 1)"
    echo "  --tp-size=<size>           Tensor model-parallel size (Default: 1)"
    echo "  --pp-size=<size>           Pipeline model-parallel size (Default: 1)"
    echo "  --zero-stage=<value>       Zero stage (Default: 0)"
    echo "  --use-flash-attn           Enable the FlashAttention calculation"
    echo
    echo "Training and learning rate configs"
    echo "  --train-iters=<value>      Total number of iterations to train over all training runs (Default: 500)"
    echo
    echo "Data and output configs"
    echo "  --base-src-path=<path>     Path to the source code"
    echo "  --base-data-path=<path>    Path to the training dataset"
    echo "  --base-output-path=<path>  Path to the directory for storing output"
    echo
    echo "Misc configs"
    echo "  --log-interval=<value>     Report loss and timing interval (Default: 50)"
    echo "  --eval-iters=<value>       Number of iterations to run for evaluation (Default: 1000)"
    echo "  --eval-interval=<value>    Interval between running evaluation on validation set (Default: 100)"
    echo
    echo "  --help                     Display this help message"
}

check_deepspeed() {
    ds_commit=d24629f4
    count=$(pip3 list | grep -c "deepspeed")

    if [ ${count} -eq 0 ]; then
        echo "Installing DeepSpeed..."
        pushd .
        # Remove obsolete download
        [ -d /tmp/DeepSpeed  ] && rm -rf /tmp/DeepSpeed
    
        git clone --recursive https://github.com/microsoft/DeepSpeed.git /tmp/DeepSpeed
        cd /tmp/DeepSpeed
        git checkout ${ds_commit}
        pip install .[dev,1bit,autotuning]
        popd
    fi
}

export CUDA_DEVICE_MAX_CONNECTIONS=1

check_deepspeed
###############################################################################
### Main configs
num_nodes=${SLURM_JOB_NUM_NODES:-1}
seqlen=1024

model_size="6.7B"
num_layers=32
hidden_size=4096
num_attn_heads=32
global_batch_size=1024
lr=1.2e-4
min_lr=1.2e-5
###############################################################################
### Parallelism and optimization configs
gpus_per_node=${gpus_per_node:-8}

## Micro batch size per data parallel group
## Make sure that MICRO_BATCH_SIZE <= GLOBAL_BATCH_SIZE*TP_SIZE*PP_SIZE/NUM_GPUS
micro_batch_size=1

## Tensor model parallelism, 1 is no TP
## Currently, MoE models have divergence issue when TP > 1
tp_size=1

## Pipeline model parallelism
## Currently, we do not support PP for MoE. To disable PP, set PP_SIZE to 1 and
## use the "--no-pipeline-parallel" flag.
pp_size=1

## ZeRO
zero_stage=0

## FlashAttention
use_flash_attn=0
###############################################################################
### Training and learning rate configs
train_iters=500

train_samples=300000
warmup_samples=3000
decay_samples=$((train_samples - warmup_samples))
###############################################################################
### Data and output configs
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base_dir="${script_dir%/*}"

base_src_path=${base_src_path:-"${base_dir}"}
base_data_path=${base_data_path:-"/dataset"}
base_output_path=${base_output_path:-"/output"}
ds_config_dir=${ds_config_dir:-"${base_dir}/ds_config"}
###############################################################################
### Misc configs
log_interval=50
eval_iters=100
eval_interval=1000
save_interval=10000

## Standard deviation for weight initialization
## We used 0.014 for 350M/1.3B dense/MoE models, and used 0.01 for 6.7B
## dense model. Usually larger model needs lower std.
init_std=0.01

## Activation checkpointing saves GPU memory, but reduces training speed
activation_checkpoint="false"

seed=12345
###############################################################################

# Parse command-line arguments
# Check for long options using the -- separator
while [[ $# -gt 0 ]]; do
    case "$1" in
        ## Main configs
        --model=*)
            model_size="${1#*=}"
            ;;
        --seqlen=*)
            seqlen="${1#*=}"
            ;;
        
        ## Parallelism and optimization configs
        --gpus-per-node=*)
            gpus_per_node="${1#*=}"
            ;;
        --batch-size=*)
            micro_batch_size="${1#*=}"
            ;;
        --tp-size=*)
            tp_size="${1#*=}"
            ;;
        --pp-size=*)
            pp_size="${1#*=}"
            ;;
        --zero-stage=*)
            zero_stage="${1#*=}"
            ;;
        --use-flash-attn)
            use_flash_attn=1
            ;;

        ## Training and learning rate configs
        --train-iters=*)
            train_iters="${1#*=}"
            ;;

        ## Data and output configs
        --base-src-path=*)
            base_src_path="${1#*=}"
            ;;
        --base-data-path=*)
            base_data_path="${1#*=}"
            ;;
        --base-output-path=*)
            base_output_path="${1#*=}"
            ;;

        ## Misc configs
        --log-interval=*)
            log_interval="${1#*=}"
            ;;
        --eval-iters=*)
            eval_iters="${1#*=}"
            ;;
        --eval-interval=*)
            eval_interval="${1#*=}"
            ;;

        --help)
            usage
            exit 0
            ;;
        *)
            echo "Invalid option: $1" >&2
            usage
            exit 1
            ;;
    esac
    shift
done

case "${model_size}" in
    125M)
        num_layers=12
        hidden_size=768
        num_attn_heads=16
        global_batch_size=256
        lr=6.0e-4
        min_lr=6.0e-5
        ;;
    350M)
        num_layers=24
        hidden_size=1024
        num_attn_heads=16
        global_batch_size=256
        lr=3.0e-4
        min_lr=3.0e-5
        ;;
    760M)
        num_layers=24
        hidden_size=1536
        num_attn_heads=16
        global_batch_size=256
        lr=2.5e-4
        min_lr=2.5e-5
        ;;
    1.3B)
        num_layers=24
        hidden_size=2048
        num_attn_heads=16
        global_batch_size=512
        lr=2.0e-4
        min_lr=2.0e-5
        ;;
    2.7B)
        num_layers=32
        hidden_size=2560
        num_attn_heads=32
        global_batch_size=512
        lr=1.6e-4
        min_lr=1.6e-5
        ;;
    6.7B)
        num_layers=32
        hidden_size=4096
        num_attn_heads=32
        global_batch_size=1024
        lr=1.2e-4
        min_lr=1.2e-5
        ;;
    13B)
        num_layers=40
        hidden_size=5120
        num_attn_heads=40
        global_batch_size=1024
        lr=1.0e-4
        min_lr=1.0e-5
        ;;
    175B)
        num_layers=96
        hidden_size=12288
        num_attn_heads=96
        global_batch_size=1536
        lr=0.6e-4
        min_lr=0.6e-5
        ;;
    *)
        echo "Model size ${model_size} is not defined" >&2
        exit 1
        ;;
esac

###############################################################################
### Dataset
data_path="${base_data_path}/oscar-gpt2_text_document"
vocab_path="${base_data_path}/gpt2-vocab.json"
merge_path="${base_data_path}/gpt2-merges.txt"
[ -d "${base_data_path}" ] || mkdir -p "${base_data_path}"

## Download dataset if necessary
if ! [ -f "${data_path}.bin" ]; then
    wget -O "${data_path}.bin" https://the-eye.eu/public/AI/pile_neox/data/BookCorpusDataset_text_document.bin
fi
if ! [ -f "${data_path}.idx" ]; then
    wget -O "${data_path}.idx" https://the-eye.eu/public/AI/pile_neox/data/BookCorpusDataset_text_document.idx
fi
if ! [ -f "${vocab_path}" ]; then
    wget -O "${vocab_path}" https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
fi
if ! [ -f "${merge_path}" ]; then
    wget -O "${merge_path}" https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
fi
###############################################################################
### Log
[ -d "${base_output_path}/log" ] || mkdir -p "${base_output_path}/log"
[ -d "${base_output_path}/checkpoint" ] || mkdir -p "${base_output_path}/checkpoint"
[ -d "${base_output_path}/tensorboard" ] || mkdir -p "${base_output_path}/tensorboard"

curr_time=$(date "+%Y-%m-%d_%H-%M-%S")
host="${HOSTNAME}"
base_name="train_gpt2-${model_size}_gpus-${gpus_per_node}_seq-${seqlen}_gbs-${global_batch_size}_mbs-${micro_batch_size}_tp-${tp_size}_pp-${pp_size}_zero-${zero_stage}"
log_name="${base_output_path}/log/${base_name}_${host}_${curr_time}.log"

checkpoint_path="${base_output_path}/checkpoint/${name}"
tensorboard_dir="${base_output_path}/tensorboard/${name}_${host}_${curr_time}"
###############################################################################
### DeepSpeed configs
[ -d "${ds_config_dir}" ] || mkdir -p "${ds_config_dir}"
ds_config_json="${config_dir}/ds_config_gbs-${global_batch_size}_mbs-${micro_batch_size}_zero-${zero_stage}.json"

## Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat << EOT > $ds_config_json
{
  "train_micro_batch_size_per_gpu": $micro_batch_size,
  "train_batch_size": $global_batch_size,
  "gradient_clipping": 1.0,
  
  "zero_optimization": {
    "stage": $zero_stage
  },
  
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },
  
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT
###############################################################################

data_options=" \
    --vocab-file ${vocab_path} \
    --merge-file ${merge_path} \
    --data-path ${data_path} \
    --data-impl mmap"

megatron_options=" \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --tensor-model-parallel-size ${tp_size} \
    --num-layers ${num_layers} \
    --hidden-size ${hidden_size} \
    --num-attention-heads ${num_attn_heads} \
    --micro-batch-size ${micro_batch_size} \
    --global-batch-size ${global_batch_size} \
    --seq-length ${seqlen} \
    --max-position-embeddings ${seqlen} \
    --train-iters ${train_iters} \
    --lr ${lr} \
    --lr-decay-style cosine \
    --min-lr ${min_lr} \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --lr-warmup-fraction .01 \
    --hysteresis 2 \
    --num-workers 0 \
    --seed ${seed} \
    --distributed-backend nccl \
    --fp16"

if [ ${use_flash_attn} -gt 0 ]; then
    megatron_options="${megatron_options} \
        --use-flash-attn"
fi

if [ "${activation_checkpoint}" = "true" ]; then
    megatron_options="${megatron_options} \
        --checkpoint-activations"
fi

output_options=" \
    --log-interval ${log_interval} \
    --eval-iters ${eval_iters} \
    --eval-interval ${eval_interval} \
    --save ${checkpoint_path} \
    --save-interval 1 \
    --tensorboard-dir ${tensorboard_dir} \
    --tensorboard-queue-size 1 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard"

ds_options=" \
    --deepspeed \
    --deepspeed_config ${ds_config_json}"
#    --deepspeed-activation-checkpointing"

distributed_options=" \
    --nproc_per_node=${gpus_per_node} \
    --nnodes=${num_nodes}"

if [ ${num_nodes} -gt 1 ]; then
    distributed_options=" \
        ${distributed_options} \
        --rdzv_id=${SLURM_JOB_ID} \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
fi

run_cmd="torchrun \
    ${distributed_options} \
    ${base_src_path}/pretrain_gpt.py \
    ${data_options} \
    ${megatron_options} \
    ${output_options} \
    ${ds_options}"

echo "${run_cmd}"      | tee "${log_name}"
eval "${run_cmd}" 2>&1 | tee -a "${log_name}"
