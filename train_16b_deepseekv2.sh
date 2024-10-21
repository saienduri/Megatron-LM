#!/bin/bash
set -e

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
echo $CURRENT_DIR #/myworkspace/Pai-Megatron-Patch-ROCm

echo "DLM_DATAHOME: ${DLM_DATAHOME}" #/data_dlm_2
echo "DLM_DATANAME: ${DLM_DATANAME}"  #deepseekv2-train-datasets

cd ${CURRENT_DIR}
EXPERIMENT_DIR="experiment"
mkdir -p $EXPERIMENT_DIR

DATA_DIR="${EXPERIMENT_DIR}/data/"

if [[ -n "${DLM_DATAHOME}" ]]; then
  mkdir -p $DATA_DIR
  ln -s ${DLM_DATAHOME}/ $DATA_DIR
  export DATA_DIR=$DATA_DIR/$DLM_DATAHOME
else
  export DATA_DIR
  mkdir -p $DATA_DIR
  cd $DATA_DIR
  mkdir -p deepseekv2-train-datasets
  cd deepseekv2-train-datasets
  wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/mmap_deepseekv2_datasets_text_document.bin
  wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/mmap_deepseekv2_datasets_text_document.idx
  wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/SlimPajama.json
  wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/alpaca_zh-train.json
  wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/alpaca_zh-valid.json
fi
cd ${CURRENT_DIR}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRAIN_JSON=${DATA_DIR}/deepseekv2-train-datasets/alpaca_zh-train.json
VALID_JSON=${DATA_DIR}/deepseekv2-train-datasets/alpaca_zh-valid.json
CKPT_DIR=${EXPERIMENT_DIR}/deepseek-ckpts/test_ft

MODEL_NAME=DeepSeek-V2-Lite
MODEL_SIZE=16B

MODEL=deepseek-ai/${MODEL_NAME}
TP=1
PP=1  
EP=8  
AC=sel #full
DO=true 
FL=true
SP=true
MBS=4
GBS=256 
SEQ_LEN=2048
PAD_LEN=2048
PR=bf16
SAVE_INTERVAL=None
LR=1e-5
MIN_LR=1e-6
TRAIN_ITERS=20
LR_WARMUP_ITERS=2

TRAIN_LOG=${EXPERIMENT_DIR}/MI300X-$MODEL_NAME-${PR}-seq${SEQ_LEN}-tp${TP}pp${PP}ep${EP}-mbs${MBS}gbs${GBS}-ac_${AC}-do_${DO}-fa_${FL}-sp_${SP}-${TIMESTAMP}.log

sh run_finetune_deepseekv2.sh  \
dsw \
$MODEL_SIZE \
$MBS    \
$GBS    \
$LR   \
$MIN_LR   \
$SEQ_LEN  \
$PAD_LEN  \
$PR  \
$TP   \
$PP  \
$EP \
$AC \
$DO \
$FL \
$SP \
$SAVE_INTERVAL  \
$TRAIN_JSON   \
$VALID_JSON   \
$MODEL \
$TRAIN_ITERS   \
$LR_WARMUP_ITERS   \
$CKPT_DIR \
2 >& 1 | tee ${TRAIN_LOG}

echo 'import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog="Process Log")
    parser.add_argument("filename")
    args = parser.parse_args()

    with open(args.filename) as f:
        lines = f.readlines()
    lines = lines[3:-1]
    lines = [float(a) for a in lines]
    mean = np.mean(np.array(lines))
    print(mean)' > mean_log_value.py

# echo '============================================================================================================'
grep -Eo 'throughput per GPU [^|]*' $TRAIN_LOG | sed -E 's/.*throughput per GPU \(TFLOP\/s\/GPU\): ([0-9\.]+).*/\1/' > tmp.txt
throughput=$(python3 mean_log_value.py tmp.txt)
echo "throughput per GPU: $throughput"
rm tmp.txt

# echo '============================================================================================================'
grep -Eo 'elapsed time per iteration [^|]*' $TRAIN_LOG | sed -E 's/.*elapsed time per iteration \(ms\): ([0-9\.]+).*/\1/' > tmp.txt
iteration_time=$(python3 mean_log_value.py tmp.txt)
echo "elapsed time per iteration: $iteration_time"
rm tmp.txt


CSV_FILE="perf_pyt_megatron_train_deepseekv2_16b.csv"
echo "model,performance,metric" > "$CSV_FILE"
echo "throughput_per_GPU,$throughput,throughput_per_GPU" >> "$CSV_FILE"
echo "elapsed_time_per_iteration,$iteration_time,elapsed_time_per_iteration" >> "$CSV_FILE"
mv "$CSV_FILE" $CURRENT_DIR/../"$CSV_FILE"