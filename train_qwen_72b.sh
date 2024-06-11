#! /bin/bash
set -ex
WORKSPACE_DIR="/workspace"
CUR_DIR=`pwd`

echo "Cur dir: ${CUR_DIR}"

export PYTORCH_TUNABLEOP_FILENAME=tuned/72b/full_tuned%d.csv
export PYTORCH_TUNABLEOP_TUNING=0
export PYTORCH_TUNABLEOP_ENABLED=1

EXPERIMENT_DIR="/workspace/pretrain/qwen_72B"
mkdir -p $EXPERIMENT_DIR

if [ ! -e "${EXPERIMENT_DIR}/logs" ]; then
  mkdir -p ${EXPERIMENT_DIR}/logs
fi

# Prepare requirements
pip3 install /workspace/tiktoken-0.6.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip3 install datasets nltk tiktoken

if [ ! -e "/opt/conda/envs/py_3.9/nltk_data/tokenizers" ]; then
  mkdir -p /opt/conda/envs/py_3.9/nltk_data/tokenizers
  cp -r /workspace/punkt /opt/conda/envs/py_3.9/nltk_data/tokenizers/
fi

# Setup HF env
HF_PATH='/workspace/transformers'

echo "DLM_DATAHOME: ${DLM_DATAHOME}"

if [[ -z "${DLM_DATAHOME}" ]]; then
        export HF_HOME="/workspace/nas_share"
        echo "No data provider found. Starting from clean cache.".
else    
        export HF_HOME=$DLM_DATAHOME
fi

export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p $HF_DATASETS_CACHE
export DATA_DIR=/workspace/llama2-training-book-corpus/
mkdir -p $DATA_DIR

if [ ! -e "${WORKSPACE_DIR}/qwen.tiktoken" ]; then
    echo "Downloading qwen.tiktoken..."
    wget -O ${WORKSPACE_DIR}/qwen.tiktoken https://huggingface.co/Qwen/Qwen-1_8B/resolve/main/qwen.tiktoken
fi

# Replace __HIP_PLATFORM_HCC__ with __HIP_PLATFORM_AMD__
find . -type f -name "*.cu" -exec sed -i 's/__HIP_PLATFORM_HCC__/__HIP_PLATFORM_AMD__/g' {} +

# Ignore the code error in Megatron-DeepSpeed
find . -type f -name "*.py" -exec sed -i 's/DS_UNIVERSAL_CHECKPOINT_INFO = True/DS_UNIVERSAL_CHECKPOINT_INFO = False/g' {} +

# Prepare the dataset
echo 'import argparse
from pathlib import Path
from datasets import load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, required=False, default="tmp/data",
                       help="Path to output JSON")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    dataset = load_dataset("bookcorpus", split="train")
    dataset.to_json(out_dir / "bookcorpus_megatron.json")' > prepare_bookcorpus_megatron_dataset.py

if [ ! -e "${DATA_DIR}/bookcorpus_megatron.json" ]; then
  echo "Prepare data..."
  python3 prepare_bookcorpus_megatron_dataset.py --out-dir ${DATA_DIR}
fi

if ! [ -f ${DATA_DIR}/bookcorpus_text_sentence.idx ]; then
  echo "Dataset file does not exist, creating..."
  python3 tools/preprocess_data.py --input ${DATA_DIR}/bookcorpus_megatron.json  --tokenizer-type QWenTokenizer --vocab-file ${WORKSPACE_DIR}/qwen.tiktoken --output-prefix ${DATA_DIR}/bookcorpus --workers `nproc` --split-sentences
  python3 tools/preprocess_data.py --input ${DATA_DIR}/bookcorpus_megatron.json  --tokenizer-type QWenTokenizer --vocab-file ${WORKSPACE_DIR}/qwen.tiktoken --output-prefix ${DATA_DIR}/bookcorpus --workers `nproc` --split-sentences
fi


#!/bin/bash
# This example script is contributed by external user https://github.com/nrailgun
set -ex

######################################
# Change the below configurations here
BASE_PATH=${EXPERIMENT_DIR}
DS_CONFIG=${BASE_PATH}/deepspeed.json
DATASET_1="${DATA_DIR}/bookcorpus_text_sentence"
DATASET="1 ${DATASET_1}"
CHECKPOINT_PATH=${EXPERIMENT_DIR}
TOKENIZER_PATH=${WORKSPACE_DIR}/qwen.tiktoken


GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0

# Qwen 72B
HIDDEN_SIZE=8192
FFN_HIDDEN_SIZE=24576
NUM_LAYERS=80
NUM_HEADS=64
MAX_POSITION_EMBEDDINGS=32768
NORM_EPS=1e-6

LR=3e-4
MIN_LR=3e-5
WEIGHT_DECAY=0.1
GRAD_CLIP=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

OUTPUT_CSV="qwen_72B.csv"
if [ ! -e "$OUTPUT_CSV" ]; then
  echo "GPU Number, TP/PP/DP, MSB/GPS/ZS, Sequence Length, Samples/s, Throughput TFLOP/s/GPU, Memory" > ${OUTPUT_CSV}
fi

SEQ_LENGTH=2048
TRAIN_STEPS=5

TP="${TP:-2}"
PP="${PP:-4}"
DP="${DP:-1}"
MBS="${MBS:-1}"
GBS="${GBS:-256}"

TRAIN_LOG="${EXPERIMENT_DIR}/logs/log_72B_${TP}_${PP}_${DP}_${MBS}_${GBS}_${SEQ_LENGTH}.txt"
run_cmd="torchrun $DISTRIBUTED_ARGS \
        pretrain_gpt.py \
        --use-flash-attn \
        --use-distributed-optimizer \
        --transformer-impl local \
        --no-gradient-accumulation-fusion \
        --tensor-model-parallel-size $TP \
        --pipeline-model-parallel-size $PP \
        --num-layers $NUM_LAYERS \
        --hidden-size $HIDDEN_SIZE \
        --ffn-hidden-size $FFN_HIDDEN_SIZE \
        --num-attention-heads $NUM_HEADS \
        --micro-batch-size $MBS \
        --global-batch-size $GBS \
        --seq-length $SEQ_LENGTH \
        --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
        --train-iters $TRAIN_STEPS \
        --data-path $DATASET \
        --tokenizer-type QWenTokenizer \
        --vocab-file $TOKENIZER_PATH \
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
        --adam-eps 1e-8 \
        --log-interval 1 \
        --save-interval 10000 \
        --eval-interval 1000 \
        --eval-iters 1 \
        --bf16 \
        --attention-dropout 0 \
        --hidden-dropout 0 \
        --untie-embeddings-and-output-weights \
        --use-rotary-position-embeddings \
        --disable-bias-linear \
        --add-qkv-bias-linear \
        --normalization RMSNorm \
        --swiglu \
        --no-async-tensor-model-parallel-allreduce \
        --log-throughput 2>&1 | tee $TRAIN_LOG"

echo ${run_cmd} | tee -a $TRAIN_LOG
eval ${run_cmd}

if grep -q "OutOfMemoryError" $TRAIN_LOG; then
    sleep 2
    echo "${GPUS_PER_NODE}, ${TP}/${PP}/${DP}, ${MBS}/${GBS}/${ZERO_STAGE}, ${SEQ_LENGTH}, -, -, OOM" >> ${OUTPUT_CSV}
    continue
fi

# output performance metric
loss=$(grep -Eo 'lm loss: [^|]*' $TRAIN_LOG | tail -n1 | sed 's/lm loss: //g' | awk -F"E" 'BEGIN{OFMT="%10.10f"} {print $1 * (10 ^ $2)}')

total_samples=$(grep -Eo 'consumed samples: [^|]*' $TRAIN_LOG | tail -n1 | sed 's/consumed samples: //g' | awk '{print $1}')
TIME_PER_ITERATION=$(grep -o 'elapsed time per iteration (ms): [^|]*' $TRAIN_LOG | tail -n1 | sed 's/elapsed time per iteration (ms): //g' | awk '{print $1}')
SAMPLES_PER_SECOND=$(awk "BEGIN {printf \"%.2f\", $GBS * 1000  / $TIME_PER_ITERATION}")

THROUGHPUT=$(grep -o 'throughput per GPU (TFLOP/s/GPU): [^|]*' $TRAIN_LOG | tail -n1 | sed 's#throughput per GPU (TFLOP/s/GPU): ##g' | awk '{print $1}')

MEM_USAGE=$(grep -Eo 'mem usages: [^|]*' $TRAIN_LOG | sed -E 's/.*mem usages: ([0-9\.]+).*/\1/'  | head -1)

echo "${GPUS_PER_NODE}, ${TP}/${PP}/${DP}, ${MBS}/${GBS}/${ZERO_STAGE}, ${SEQ_LENGTH}, ${SAMPLES_PER_SECOND}, ${THROUGHPUT}, ${MEM_USAGE}" >> ${OUTPUT_CSV}