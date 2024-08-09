# MASTER_ADDR=10.11.8.150
# MASTER_PORT=10097

# torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT debug_gloo.py

NNODES=5
RANK=4

NUM_GROUPS=$(( ${NNODES} - 1 ))
if [[ $RANK -eq $NUM_GROUPS ]]; then
    echo 6666
else
        echo "Not the final node; check another the output for another node!"
        exit 1
fi

