#!/bin/bash
set -ex
WORKSPACE_DIR="./workspace"
CUR_DIR=`pwd`

echo "Current directory: ${CUR_DIR}"

EXPERIMENT_DIR="tmp"
mkdir -p $EXPERIMENT_DIR
mkdir -p $WORKSPACE_DIR

export DLM_SYSTEM_GPU_ARCHITECTURE="gfx942"


# Setup HF env
HF_PATH='./workspace/transformers'

echo "DLM_DATAHOME: ${DLM_DATAHOME}"

if [ -z "${DLM_DATAHOME}" ]; then
        export HF_HOME="./workspace/nas_share"
        echo "No data provider found. Starting from clean cache.".
else    
        export HF_HOME=$DLM_DATAHOME
fi

export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p $HF_DATASETS_CACHE
export DATA_DIR=$EXPERIMENT_DIR/data
mkdir -p $DATA_DIR

# Replace __HIP_PLATFORM_HCC__ with __HIP_PLATFORM_AMD__
find . -type f -name "*.cu" -exec sed -i 's/__HIP_PLATFORM_HCC__/__HIP_PLATFORM_AMD__/g' {} +

# Ignore the code error in Megatron-DeepSpeed
find . -type f -name "*.py" -exec sed -i 's/DS_UNIVERSAL_CHECKPOINT_INFO = True/DS_UNIVERSAL_CHECKPOINT_INFO = False/g' {} +

# Prepare the dataset
echo 'import argparse
import os
from pathlib import Path
from datasets import load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, required=False, default="tmp/data",
                       help="Path to output JSON")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    if not os.path.exists(out_dir / "bookcorpus_megatron.json"):

      dataset = load_dataset("bookcorpus", split="train")
      dataset.to_json(out_dir / "bookcorpus_megatron.json")' > prepare_bookcorpus_megatron_dataset.py

# check tokenizer in preprocess_data.py
python3 prepare_bookcorpus_megatron_dataset.py --out-dir ${DATA_DIR}
if ! [ -f ${DATA_DIR}/bookcorpus_text_sentence.idx ]; then
  echo "Dataset file does not exist, creating..."
  python3 tools/preprocess_data.py --input ${DATA_DIR}/bookcorpus_megatron.json  --tokenizer-type DeepSeekV2Tokenizer --output-prefix ${DATA_DIR}/bookcorpus --workers `nproc` --split-sentences
  python3 tools/preprocess_data.py --input ${DATA_DIR}/bookcorpus_megatron.json  --tokenizer-type DeepSeekV2Tokenizer --output-prefix ${DATA_DIR}/bookcorpus --workers `nproc` --split-sentences
fi

#!/bin/bash
# This example script is contributed by external user https://github.com/nrailgun
set -ex

######################################
# Change the below configurations here
BASE_PATH=${EXPERIMENT_DIR}
DS_CONFIG=${BASE_PATH}/deepspeed.json
DATASET_1="./${DATA_DIR}/bookcorpus_text_sentence"
DATASET="1 ${DATASET_1}"
CHECKPOINT_PATH=${EXPERIMENT_DIR}
MODEL_NAME="deepseek-ai/DeepSeek-V2-Lite"
# TOKENIZER_PATH=${EXPERIMENT_DIR}/tokenizer.model # offical llama tokenizer.model
# Need to check tokenizer

TP=1
PP=1
ZERO_STAGE=1

GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0

HIDDEN_SIZE=2048 #5120
FFN_HIDDEN_SIZE=10944 #12288
NUM_LAYERS=27 #60
NUM_HEADS=16 #128
SEQ_LENGTH=2048  #unknown
MAX_POSITION_EMBEDDINGS=163840 #163840 
NUM_KV_HEADS=16 #128

MICRO_BATCH_SIZE=6
GLOBAL_BATCH_SIZE=48 # e.g. llama: 4M tokens
TRAIN_STEPS=200 # e.g. llama: 1T tokens / 4M tokens_per_batch = 250000 steps
LR=3e-4
MIN_LR=3e-5
# LR_WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

## Activation checkpointing saves GPU memory, but reduces training speed
# activation_checkpoint="true"
activation_checkpoint="false"

# Set to cpu for offloading to cpu for larger models
# OFFLOAD_DEVICE="cpu"
OFFLOAD_DEVICE="none"
# CPU_OPTIM=" --cpu-optimizer"
CPU_OPTIM=""

cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "steps_per_print": 1,
  "zero_optimization": {
    "stage": $ZERO_STAGE,
    "offload_optimizer": {
      "device": "$OFFLOAD_DEVICE",
      "buffer_count": 4,
      "pin_memory": true
    },
    "offload_param": {
      "device": "$OFFLOAD_DEVICE",
      "buffer_count": 4,
      "pin_memory": true
    },
    "stage3_max_live_parameters": 3e9,
    "stage3_max_reuse_distance": 3e9,
    "stage3_param_persistence_threshold": 1e5,
    "stage3_prefetch_bucket_size": 5e7,
    "contiguous_gradients": true,
    "reduce_bucket_size": 90000000,
    "sub_group_size": 1e9
  },
  "bf16": {
    "enabled": true
  }
}
EOT

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --no-pipeline-parallel ${ds_args}" 
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"

if [ "${activation_checkpoint}" = "true" ]; then
  ds_args="--deepspeed-activation-checkpointing ${ds_args}"
  ds_args="--partition-activations ${ds_args}"
  ds_args="--contigious-checkpointing ${ds_args}"
  ds_args="--checkpoint-in-cpu ${ds_args}"

  ## old argument for recomputing the transformer layer
  # ds_args="--checkpoint-activations ${ds_args}"

  ## new argument for recomputing the transformer layer
  ds_args="--recompute-granularity full --recompute-method uniform ${ds_args}"
  ## new argument for recomputing only the attention layer
  # ds_args="--recompute-granularity selective ${ds_args}"
fi

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# Adjust tokenizer parameters
# what is split here?
run_cmd="torchrun $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --use-flash-attn-v2 \
       --no-gradient-accumulation-fusion \
       --tensor-model-parallel-size $TP \
       --pipeline-model-parallel-size $PP \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --ffn-hidden-size $FFN_HIDDEN_SIZE \
       --num-attention-heads $NUM_HEADS \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --seq-length $SEQ_LENGTH \
       --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
       --train-iters $TRAIN_STEPS \
       --save $CHECKPOINT_PATH \
       --tokenizer_model $MODEL_NAME \
       --data-path $DATASET \
       --data-impl mmap \
       --tokenizer-type DeepSeekV2Tokenizer \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr $LR \
       --lr-decay-style cosine \
       --min-lr $MIN_LR \
       --weight-decay $WEIGHT_DECAY \
       --clip-grad $GRAD_CLIP \
       --lr-warmup-fraction 0.01 \
       --optimizer adam \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --bf16 \
       --no-query-key-layer-scaling \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --use-rotary-position-embeddings \
       --untie-embeddings-and-output-weights \
       --swiglu \
       --normalization rmsnorm \
       --disable-bias-linear \
       --num-key-value-heads $NUM_KV_HEADS \
       $CPU_OPTIM $ds_args 2>&1 | tee log.txt"

echo ${run_cmd}
eval ${run_cmd}

# output performance metric
train_loss=$(grep -Eo 'lm loss: [^|]*' log.txt | tail -n1 | sed 's/lm loss: //g' | awk -F"E" 'BEGIN{OFMT="%10.10f"} {print $1 * (10 ^ $2)}')
train_perf=$(grep -Eo 'samples per second: [^|]*' log.txt | tail -n1 | sed 's/samples per second: //g' )

set +x
CSV_RESULTS="${CUR_DIR}/../llama2_7b_training.csv"
echo "model,performance,metric" > ${CSV_RESULTS}
echo "loss,${train_loss},loss" >> ${CSV_RESULTS}
echo "perf,${train_perf},samples_per_second" >> ${CSV_RESULTS}