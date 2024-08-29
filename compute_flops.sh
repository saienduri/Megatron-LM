#!/bin/bash

# set -x


# parsing input arguments
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)
   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"
   export "$KEY"="$VALUE"
done

TRAIN_LOG="${TRAIN_LOG:-}"
MODEL_SIZE="${MODEL_SIZE:-70}"
TP="${TP:-8}"
PP="${PP:-1}"
MBS="${MBS:-2}"
BS="${BS:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
NNODES="${NNODES:-1}"
EXP_NAME="${EXP_NAME:-perf}"
GPUS_PER_NODE=`python3 -c "import torch; print(torch.cuda.device_count())"`
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# echo '============================================================================================================'
grep -Eo 'throughput per GPU [^|]*' $TRAIN_LOG | sed -E 's/.*throughput per GPU \(TFLOP\/s\/GPU\): ([0-9\.]+).*/\1/' > tmp.txt
PERFORMANCE=$(python3 mean_log_value.py tmp.txt)
echo "throughput per GPU: $PERFORMANCE" |& tee -a $TRAIN_LOG
rm tmp.txt

# echo '============================================================================================================'
grep -Eo 'elapsed time per iteration [^|]*' $TRAIN_LOG | sed -E 's/.*elapsed time per iteration \(ms\): ([0-9\.]+).*/\1/' > tmp.txt
ETPI=$(python3 mean_log_value.py tmp.txt)
echo "elapsed time per iteration: $ETPI" |& tee -a $TRAIN_LOG

TIME_PER_ITER=$(python3 mean_log_value.py tmp.txt 2>/dev/null | awk '{printf "%.6f", $0}')
TGS=$(awk -v bs="$BS" -v sl="$SEQ_LENGTH" -v tpi="$TIME_PER_ITER" -v ws="$WORLD_SIZE" 'BEGIN {printf "%.6f", bs * sl * 1000/ (tpi * ws)}')
echo "tokens/GPU/s: $TGS" |& tee -a $TRAIN_LOG
rm tmp.txt

echo '============================================================================================================'
grep -Eo 'mem usages: [^|]*' $TRAIN_LOG | sed -E 's/.*mem usages: ([0-9\.]+).*/\1/' > tmp.txt
MEMUSAGE=$(python3 mean_log_value.py tmp.txt)
echo "mem usages: $MEMUSAGE" |& tee -a $TRAIN_LOG
rm tmp.txt




'EXP_NAME	#Nodes	Model 	Seq Len	Micro batch	Global Batch	TP	PP	Tokens/Sec/GPU	TFLOPs/s/GPU	Memory Usage'
echo "${EXP_NAME}	$NNODES	$MODEL_SIZE	$SEQ_LENGTH	$MBS	$BS	$TP	$PP	$TGS	$PERFORMANCE	$MEMUSAGE	$ETPI" |& tee -a ../out.csv
echo "${EXP_NAME}	$NNODES	$MODEL_SIZE	$SEQ_LENGTH	$MBS	$BS	$TP	$PP	$TGS	$PERFORMANCE	$MEMUSAGE	$ETPI" |& tee -a out.csv

