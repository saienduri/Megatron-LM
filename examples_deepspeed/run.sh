#!/bin/bash

cmd="HIP_FORCE_DEV_KERNARG=1 GLOBAL_BATCH_SIZE=3072 MICRO_BATCH_SIZE=24 TP_SIZE=1 PP_SIZE=1 bash /root/Megatron-Deepspeed-PoC/examples_deepspeed/pretrain_gpt_with_mp.sh"

echo ${cmd}
eval ${cmd}
