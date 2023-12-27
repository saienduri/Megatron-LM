#!/bin/bash

set -ex

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base_dir="${script_dir%/*}"

###############################################################################
### Main configs
SEQ_LEN=1024

## GPT-3 Large 760M

#MODEL_SIZE="760m"
#NUM_LAYERS=24
#HIDDEN_SIZE=1536
#NUM_ATTN_HEADS=16
#GLOBAL_BATCH_SIZE=256
#LR=2.5e-4
#MIN_LR=2.0e-5

MODEL_SIZE="6.7B"
NUM_LAYERS=32
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
#GLOBAL_BATCH_SIZE=1024
LR=1.2e-4
MIN_LR=1.2e-5

#MODEL_SIZE="125M"
#NUM_LAYERS=12
#HIDDEN_SIZE=768
#NUM_ATTN_HEADS=16
#GLOBAL_BATCH_SIZE=256
#LR=6.0e-4
#MIN_LR=6.0e-5

###############################################################################
### Parallelism configs
NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
NODE_RANK=${SLURM_NODEID:-0}
GPUS_PER_NODE=8

## Micro batch size per data parallel group
## Make sure that MICRO_BATCH_SIZE <= GLOBAL_BATCH_SIZE*TP_SIZE*PP_SIZE/NUM_GPUS
#MICRO_BATCH_SIZE=8

## Tensor model parallelism, 1 is no TP
## Currently, MoE models have divergence issue when TP > 1
#TP_SIZE=2

## Pipeline model parallelism
## Currently, we do not support PP for MoE. To disable PP, set PP_SIZE to 1 and
## use the "--no-pipeline-parallel" flag.
#PP_SIZE=1

## ZeRO
ZERO_STAGE=1
###############################################################################
### Training and learning rate configs
TRAIN_ITERS=50

TRAIN_SAMPLES=300000
WARMUP_SAMPLES=3000
DECAY_SAMPLES=$((TRAIN_SAMPLES - WARMUP_SAMPLES))

## Another termination condition in minutes. Set it large enough to avoid
## undesired early termination.
EXIT_DURATION=30
###############################################################################
### Data and output configs
VOCAB_FILE="${VOCAB_FILE:-"/megatron/Megatron-DeepSpeed/dataset/gpt2-vocab.json"}"
MERGE_FILE="${MERGE_FILE:-"/megatron/Megatron-DeepSpeed/dataset/gpt2-merges.txt"}"
#DATA_PATH="${DATA_PATH:-"/megatron/Megatron-DeepSpeed/dataset/BookCorpusDataset_text_document"}"
DATA_PATH="${DATA_PATH:-"/megatron/Megatron-DeepSpeed/dataset/my-gpt2_text_document"}"

curr_time=$(date "+%Y-%m-%d_%H-%M-%S")
host="${HOSTNAME}"
name="train_gpt_${MODEL_SIZE}_gbs-${GLOBAL_BATCH_SIZE}_mbs-${MICRO_BATCH_SIZE}_gpus-${NUM_GPUS}_mp-${MP_SIZE}_pp-${PP_SIZE}"
base_output_dir="${base_dir}/output"

[ -d "${base_output_dir}/checkpoint" ] || mkdir -p "${base_output_dir}/checkpoint"
[ -d "${base_output_dir}/tensorboard" ] || mkdir -p "${base_output_dir}/tensorboard"
[ -d "${base_output_dir}/log" ] || mkdir -p "${base_output_dir}/log"

CHECKPOINT_PATH="${base_output_dir}/checkpoint/${name}"
TENSORBOARD_DIR="${base_output_dir}/tensorboard/${name}_${host}_${curr_time}"
LOG="${base_output_dir}/log/${name}_${host}_${curr_time}.log"
###############################################################################
### Misc configs
LOG_INTERVAL=10
EVAL_ITERS=10
EVAL_INTERVAL=100
SAVE_INTERVAL=1000

## Standard deviation for weight initialization
## We used 0.014 for 350M/1.3B dense/MoE models, and used 0.01 for 6.7B
## dense model. Usually larger model needs lower std.
INIT_STD=0.01

## Activation checkpointing saves GPU memory, but reduces training speed
ACTIVATION_CHECKPOINT="false"

SEED=${RANDOM}
###############################################################################
### DeepSpeed configs
ds_config_dir="${base_dir}/ds_config"
ds_config_json="${config_dir}/ds_config_gbs-${GLOBAL_BATCH_SIZE}_mbs-${MICRO_BATCH_SIZE}_zero-${ZERO_STAGE}.json"

[ -d "${ds_config_dir}" ] || mkdir -p "${ds_config_dir}"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat << EOT > $ds_config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "train_batch_size": $GLOBAL_BATCH_SIZE,
  "gradient_clipping": 1.0,
  
  "zero_optimization": {
    "stage": $ZERO_STAGE
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
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-path $DATA_PATH \
    --data-impl mmap"

megatron_options=" \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_ATTN_HEADS \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --train-iters $TRAIN_ITERS \
    --split 995,5,0 \
    --lr $LR \
    --min-lr $MIN_LR \
    --lr-decay-style cosine \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --lr-warmup-fraction .01 \
    --log-interval $LOG_INTERVAL \
    --hysteresis 2 \
    --num-workers 0 \
    --seed $SEED \
    --distributed-backend nccl \
    --fp16 \
    --no-gradient-accumulation-fusion \
    --use-flash-attn-v1"

if [ "${ACTIVATION_CHECKPOINT}" = "true" ]; then
    megatron_options="$megatron_options \
        --checkpoint-activations"
fi

output_options=" \
    --log-interval $LOG_INTERVAL \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --tensorboard-dir $TENSORBOARD_DIR \
    --tensorboard-queue-size 1 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard"

ds_options=" \
    --deepspeed \
    --deepspeed_config $ds_config_json"
#    --deepspeed-activation-checkpointing"


distributed_options=" \
    --nproc_per_node=${GPUS_PER_NODE} \
    --nnodes=${NUM_NODES}"

if [ ${NUM_NODES} -gt 1 ]; then
    distributed_options=" \
        ${distributed_options} \
        --rdzv_id=12345 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
fi

run_cmd="torchrun \
    $distributed_options \
    $base_dir/pretrain_gpt.py \
    $data_options \
    $megatron_options \
    $output_options \
    $ds_options 2>&1"

echo "${run_cmd}" | tee "$LOG"
eval "${run_cmd}" | tee -a "$LOG"
