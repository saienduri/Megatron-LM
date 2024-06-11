#!/bin/bash


#SBATCH <SLURM OPTIONS> --nodes=128 --exclusive --ntasks-per-node=8 --job-name=megatron_gpt3_175b


DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs


## TODO: Follow README to prepare your dataset

# DATASET_1="<PATH TO THE FIRST DATASET>"
# DATASET_2="<PATH TO THE SECOND DATASET>"
# DATASET_3="<PATH TO THE THIRD DATASET>"
# DATASET="0.2 ${DATASET_1} 0.3 ${DATASET_2} 0.5 ${DATASET_3}"

VOCAB="gpt2-vocab.json"
MERGE="gpt2-merges.txt"
CKPT_DIR="outputs/gpt3_13b/checkpoints"
TENSORBOARD_DIR="outputs/gpt3_13b/tensorboard"
DATASET_1="my-gpt2_text_document"
# DATASET_1="bookcorpus_text_sentence"
DATASET="1 ${DATASET_1}"

# export CUDA_DEVICE_MAX_CONNECTIONS=1


TP=1
PP=1
MICRO_BATCH_SIZE=6
GLOBAL_BATCH_SIZE=240

options=" \
	--use-distributed-optimizer \
	--use-flash-attn \
	--log-throughput\
	--no-gradient-accumulation-fusion \
	--no-async-tensor-model-parallel-allreduce \
	--tensor-model-parallel-size ${TP} \
	--pipeline-model-parallel-size ${PP} \
        --num-layers 40 \
        --hidden-size 5120 \
        --num-attention-heads 40 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
	--micro-batch-size ${MICRO_BATCH_SIZE} \
	--global-batch-size ${GLOBAL_BATCH_SIZE} \
	--train-samples 146484375 \
       	--lr-decay-samples 126953125 \
        --lr-warmup-samples 183105 \
        --lr 6.0e-5 \
	--min-lr 1.0e-6 \
        --lr-decay-style cosine \
        --log-interval 1 \
        --eval-iters 40 \
        --eval-interval 1000 \
	--data-path ${DATASET} \
	--vocab-file ${VOCAB} \
	--merge-file ${MERGE} \
	--save-interval 1000 \
	--save ${CKPT_DIR} \
	--load ${CKPT_DIR} \
	--split 98,2,0 \
	--clip-grad 1.0 \
	--weight-decay 0.1 \
	--adam-beta1 0.9 \
	--adam-beta2 0.95 \
	--init-method-std 0.005 \
	--tensorboard-dir ${TENSORBOARD_DIR} \
	--bf16 "

log_file=$DIR/logs/gpt3-13b_$DATETIME.log 

GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
torchrun $DISTRIBUTED_ARGS ${DIR}/pretrain_gpt.py $@ ${options} | tee $log_file

set +x
