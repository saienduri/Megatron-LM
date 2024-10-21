#!/bin/bash
set -ex

export GPU_MAX_HW_QUEUES=2
export TORCH_NCCL_HIGH_PRIORITY=1

# parsing input arguments
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"
   export "$KEY"="$VALUE"
done

# Change for multinode config
export CUDA_DEVICE_MAX_CONNECTIONS=1

TEE_OUTPUT="${TEE_OUTPUT:-1}"
NO_TORCH_COMPILE="${NO_TORCH_COMPILE:-0}"
USE_FLASH_ATTN="${USE_FLASH_ATTN:-1}"
NO_TRAINING="${NO_TRAINING:-0}" # NO_TRAINING=1: for computing metrics only
ENABLE_PROFILING="${ENABLE_PROFILING:-0}"
ENABLE_ROPE="${ENABLE_ROPE:-1}"
ENABLE_ROPE_TE="${ENABLE_ROPE_TE:-1}"
ENABLE_MOCK_DATA="${ENABLE_MOCK_DATA:-1}"
DUMMY_RUN="${DUMMY_RUN:-0}"
ADD_TASK="${ADD_TASK:-0}"
LABEL="${LABEL:-"test"}"
LOG_DIR="profile/${LABEL}"
echo "NO_TRAINING=$NO_TRAINING"

CWD=`pwd`
GPUS_PER_NODE=`python -c "import torch; print(torch.cuda.device_count())"`

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=23731
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

MODEL_SIZE="${MODEL_SIZE:-70}"
TP="${TP:-8}"
PP="${PP:-1}"
MBS="${MBS:-2}"
BS="${BS:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
TOTAL_ITERS="${TOTAL_ITERS:-4}"
SEQ_PARALLEL="${SEQ_PARALLEL:-1}" 
CONTI_PARAMS="${CONTI_PARAMS:-0}"
OPTIMIZER="${OPTIMIZER:-sgd}"
TE_FP16="${TE_FP16:-1}"

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

if ! [ "$ENABLE_MOCK_DATA" -eq 1 ]; then
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

      dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
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

if [ "$TE_FP16" -eq 1 ]; then
    TRAIN_LOG="${EXPERIMENT_DIR}/train_${MODEL_SIZE}B_iter${TOTAL_ITERS}_mbs${MBS}_bs${BS}_tp${TP}_pp${PP}_seq${SEQ_LENGTH}_optim_${OPTIMIZER}_nocompile${NO_TORCH_COMPILE}_fa_${USE_FLASH_ATTN}_seqpara_${SEQ_PARALLEL}_contiparam_${CONTI_PARAMS}_TE_FP16_${LABEL}.log"
else
    TRAIN_LOG="${EXPERIMENT_DIR}/train_${MODEL_SIZE}B_iter${TOTAL_ITERS}_mbs${MBS}_bs${BS}_tp${TP}_pp${PP}_seq${SEQ_LENGTH}_optim_${OPTIMIZER}_nocompile${NO_TORCH_COMPILE}_fa_${USE_FLASH_ATTN}_seqpara_${SEQ_PARALLEL}_contiparam_${CONTI_PARAMS}_${LABEL}.log"
fi

ZERO_STAGE=1

if [[ $MODEL_SIZE -eq 7 ]]; then
        HIDDEN_SIZE=4096 # e.g. llama-13b: 5120
        FFN_HIDDEN_SIZE=11008 # e.g. llama-13b: 13824
        NUM_LAYERS=32 # e.g. llama-13b: 40
        NUM_HEADS=32 # e.g. llama-13b: 40
        SEQ_LENGTH=$SEQ_LENGTH
        MAX_POSITION_EMBEDDINGS=$MAX_POSITION_EMBEDDINGS
        NUM_KV_HEADS=32 # llama2 70B uses GQA
elif [[ $MODEL_SIZE -eq 16 ]]; then
        HIDDEN_SIZE=2048 # e.g. llama-13b: 5120
        FFN_HIDDEN_SIZE=10944 # e.g. llama-13b: 13824 (intermediate_size)
        NUM_LAYERS=27 # e.g. llama-13b: 40
        NUM_HEADS=16 # e.g. llama-13b: 40
        SEQ_LENGTH=$SEQ_LENGTH
        MAX_POSITION_EMBEDDINGS=$MAX_POSITION_EMBEDDINGS
        NUM_KV_HEADS=16 # llama2 70B uses GQA
else
        echo "Model size not supported."
        exit 1
fi

PROFILING_DIR="${EXPERIMENT_DIR}/perf_${MODEL_SIZE}B_iter${TOTAL_ITERS}_mbs${MBS}_bs${BS}_tp${TP}_pp${PP}_seq${SEQ_LENGTH}_optim_${OPTIMIZER}_nocompile${NO_TORCH_COMPILE}_fa_${USE_FLASH_ATTN}_seqpara_${SEQ_PARALLEL}_contiparam_${CONTI_PARAMS}"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NUM_HEADS \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --swiglu \
    --init-method-std 0.02 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --micro-batch-size $MBS \
    --global-batch-size $BS \
    --lr 3.0e-4 \
    --train-iters $TOTAL_ITERS \
    --lr-decay-style cosine \
    --min-lr 3.0e-5 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction .01 \
    --optimizer $OPTIMIZER \
    --no-async-tensor-model-parallel-allreduce \
    --clip-grad 1.0 \
    --bf16 \
    --no-masked-softmax-fusion \
    --overlap-grad-reduce \
"

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

DATA_ARGS="
    --tokenizer-type DeepSeekV2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --split 949,50,1 \
"
OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --log-throughput \
    --no-save-optim \
    --eval-iters -1
"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
EXTRA_ARGS="
    --group-query-attention \
    --num-query-groups $NUM_GROUPS \
    --no-gradient-accumulation-fusion \
    --distributed-backend nccl \
    --distributed-timeout-minutes 30
"

if [ "$ENABLE_PROFILING" -eq 1 ]; then
EXTRA_ARGS="$EXTRA_ARGS --profile --use-pytorch-profiler --tensorboard-dir $LOG_DIR"
fi

if [ "$ADD_TASK" -eq 1 ]; then
EXTRA_ARGS="$EXTRA_ARGS --task gpt_chat"
fi
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
       --use-flash-attn \
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
       --data-path $DATASET \
       --data-impl mmap \
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
       --normalization RMSNorm \
       --disable-bias-linear \
       --num-key-value-heads $NUM_KV_HEADS \
       --init-method-std 0.02 \
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