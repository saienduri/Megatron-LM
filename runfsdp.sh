export GPU_MAX_HW_QUEUES=2
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_CHECKS_DISABLE=1
export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=0
export NCCL_SOCKET_IFNAME=ens51f0np0
export GLOO_SOCKET_IFNAME=ens51f0np0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_PROTO=Simple
export RCCL_MSCCL_ENABLE=0
export TOKENIZERS_PARALLELISM=false
# export AMD_LOG_LEVEL=3
# export AMD_SERIALIZE_KERNEL=3
export HSA_NO_SCRATCH_RECLAIM=1

# parsing input arguments
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-23731}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"

# torchrun --nnodes=4 --node_rank=1 --master_addr=tw038 --master_port=57789 --nproc-per-node=8 fsdp.py -b 5
# torchrun --nnodes=2 --node_rank=1 --master_addr=tw038 --master_port=37791 --nproc-per-node=8 fsdp.py -b 5
# torchrun --nnodes=1 --master_port=37791 --nproc-per-node=8 fsdp.py -b 5
# torchrun --nnodes=2 --node_rank=1 --master_addr=tw038 --master_port=37791 --nproc-per-node=8 fsdp.py -b 4
# torchrun --nnodes=4 --node_rank=1 --master_addr=tw038 --master_port=57789 --nproc-per-node=8 fsdp.py -b 4
torchrun --nnodes $NNODES  --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT --nproc-per-node=8 fsdp.py -b 4
