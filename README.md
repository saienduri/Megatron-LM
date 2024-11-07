# How to run
## Environment Setup

### Docker and library
#### MI300
- Pull the `rocm/pytorch-private:exec_dash_pretuned_nightly_inai_FA_ck_v0.1.1_bf16_rtn_TE_GQA` docker image. 
- Example:
   <pre>docker run -it --device /dev/dri --device /dev/kfd --network host --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged    -v  $HOME/.ssh:/root/.ssh  --shm-size 128G --name llama-70b-training-gl  rocm/pytorch-private:exec_dash_pretuned_nightly_inai_FA_ck_v0.1.1_bf16_rtn_TE_GQA /bin/bash
   </pre>

#### H100/H200
- Pull the `nvcr.io/nvidia/pytorch:24.08-py3` docker image and install some packages. 
   <pre>docker run --gpus all -it -v $HOME:$HOME  --network host  --shm-size 64G --rm nvcr.io/nvidia/pytorch:24.08-py3
   pip install netifaces
   pip install setuptools==69.5.1
   pip install transformers
   </pre>


### Network configuration
Before run the training, we need to adapt the network interface.
- Network interface
   - Currently, we are using network interface `ens51f0np0` in lines-{11,12} at our script [train_acc_loss_llama3.sh](./train_acc_loss_llama3.sh).
   - We can reset `ens51f0np0` to the network interface of the used server; `ifconfig` can list all available network interfaces.
- Infini-Band
   - For single-node test, we don't need Infini-Band, please comment out lines-{8,9}; 
   - For multi-node training, we need to configure the `NCCL_IB_HCA` and `NCCL_IB_GID_INDEX` env vars in 

### Prepare the dataset:
You can either use fake data or real data if only focusing on the throughputs.
#### Fake data
You can also use fake data for some quick test, though it leads to slower performance:
- replace line 169 in [train_acc_loss_llama3.sh](./train_acc_loss_llama3.sh) with `    --mock-data \`
#### Real data
- Install `git-lfs` inside the docker: `sudo apt update && sudo apt install git`
-Process the dataset file:
   ```
   PATH_TO_DATASET=../data
   mkdir -p $PATH_TO_DATASET
   git clone https://huggingface.co/datasets/teknium/OpenHermes-2.5 $PATH_TO_DATASET
   python openhermes_2_5_to_jsonl.py --data_file=$PATH_TO_DATASET/openhermes2_5.json
   ```
- After processing, there will be a `openhermes2_5.jsonl` file in $PATH_TO_DATASET.

- Modify the data path to `$PATH_TO_DATASET/openhermes2_5.jsonl` in line-72 at our script [train_acc_loss_llama3.sh](./train_acc_loss_llama3.sh).

### Prepare the tokenizer:

Download the correpsonding tokenizer from huggingface. 

- For example, for llama3-8B model, download the repo from [Hugging-face](https://huggingface.co/meta-llama/Meta-Llama-3-8B/tree/main)
  > Note: for Llama3, we may need to move the tokenizer out of the `original` folder.

- Assume the tokenizer is stored at TOKENIZER_MODEL

- Modify the tokenizer path to `$TOKENIZER_MODEL` in line-73 at our script [train_acc_loss_llama3.sh](./train_acc_loss_llama3.sh).

## Running the benchmarking


### Single node with performance tuning
Set the parameters in [train_acc_loss_llama3.sh](./train_acc_loss_llama3.sh) or in the bash command as follows:
- Llama3-70B with 2K sequence length
<pre>
bash train_acc_loss_llama3.sh MODEL_SIZE=70 MBS=9 BS=72 TP=8 PP=1 SEQ_LENGTH=2048
</pre>

- Llama3-8B with 2K sequence length
<pre>
bash train_acc_loss_llama3.sh MODEL_SIZE=8 MBS=7 BS=448 TP=1 PP=1 SEQ_LENGTH=2048
</pre>

- Llama3-8B with 128K sequence length
<pre>
bash train_acc_loss_llama3.sh MODEL_SIZE=8 MBS=1 BS=1 TP=8 PP=1 SEQ_LENGTH=128000
</pre>
> Note, 128K requires many resources for data processing; You can modify the ds_works in line 91 but it will lower the performance.


### Multiple node
- Launch the docker on each of the node

- Run exactly the same setup (docker, python-environment) in every node

- Set the parameters in [train_acc_loss_llama3.sh](./train_acc_loss_llama3.sh)

- For each node, modify the following lines in [train_acc_loss_llama3.sh](./train_acc_loss_llama3.sh)
   <pre>
   7: export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7 -->set to the Infini-band interface
   8: export NCCL_IB_GID_INDEX=3 --> set to the Infini-band gid_index
   11: export GLOO_SOCKET_IFNAME=ens51f0np0 --> to the network interface on the server [can by obtainted by run ifconfig]
   12: export NCCL_SOCKET_IFNAME=ens51f0np0 --> to the network interface on the server [can by obtainted by run ifconfig]
   </pre>



- For example, we want to run a four-node test with Llama3-8B with 2K sequence length; on each node:
   <pre>
   Node0: bash train_acc_loss_llama3.sh MODEL_SIZE=8 MBS=7 BS=448 TP=1 PP=1 SEQ_LENGTH=2048 MASTER_ADDR=IP_NODE0 NNODES=4 NODE_RANK=0
   Node1: bash train_acc_loss_llama3.sh MODEL_SIZE=8 MBS=7 BS=448 TP=1 PP=1 SEQ_LENGTH=2048 MASTER_ADDR=IP_NODE0 NNODES=4 NODE_RANK=1
   Node2: bash train_acc_loss_llama3.sh MODEL_SIZE=8 MBS=7 BS=448 TP=1 PP=1 SEQ_LENGTH=2048 MASTER_ADDR=IP_NODE0 NNODES=4 NODE_RANK=2
   Node3: bash train_acc_loss_llama3.sh MODEL_SIZE=8 MBS=7 BS=448 TP=1 PP=1 SEQ_LENGTH=2048 MASTER_ADDR=IP_NODE0 NNODES=4 NODE_RANK=3
   </pre>


## Fine-tuning
`pip install accelerate`

add the following after line 1245 in `./megatron/training/arguments.py`:
 ```
    group.add_argument('--gradient_accumulation_fusion', default=False,
                       help='fusing gradient accumulation to weight '
                       'gradient computation of linear layers',)
```

### Prepare model checkpoints
- Download the correpsonding checkpoints from huggingface. 
   - For example, for llama2-7B-HF model, download the repo from [Hugging-face](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- Convert it into the Megatron format; modify the following vars in `convert_hf.sh`
   ```
      HF_FORMAT_DIR=checkpoint/llama2-7b-hf         --> where we download the Huggingface checkpoint
      MEGATRON_FORMAT_DIR=checkpoint/llama2-7b-mgm  -->  where we we want to store the megatron checkpoint
      TOKENIZER_MODEL=checkpoint/llama2-7b-hf       --> where we download the Huggingface checkpoint
      TP=1 --> number of tensor parellism
   ```
### Prepare dataset
- Download the cleaned Alpaca dataset from [Hugging-face](https://huggingface.co/datasets/yahma/alpaca-cleaned/tree/main)
- Convert it into our desired format
  - `python3 alpaca_to_jsonl.py --input_file $DATA_DIR/alpaca_data_cleaned.json`
  - You will see a file `$DATA_DIR/alpaca_data_cleaned.jsonl` after convertions

### Launching the fine-tuning
- modify the following vars in `sft_llama2.sh`
   ```
   DATA_PATH=../data/alpaca_data_cleaned.jsonl  --> where we store the process data file 
   TOKENIZER_MODEL=../checkpoint/llama2-7b-hf   --> where we store the hugging-face face checkpoint
   CHECKPOINT_PATH=../checkpoint/llama2-7b-mgm  --> where we store the converted Megatron checkpoints; same as $MEGATRON_FORMAT_DIR in the above
   ```

   `HIP_VISIBLE_DEVICES=0 bash sft_llama2.sh MODEL_SIZE=7 MBS=4 BS=4`


#### If meet the following errors:
   ```
   [rank1]: 	size mismatch for embedding.word_embeddings.weight: copying a param with shape torch.Size([32000, 4096]) from checkpoint, the shape in current model is torch.Size([32128, 4096]).
   [rank1]: 	size mismatch for output_layer.weight: copying a param with shape torch.Size([32000, 4096]) from checkpoint, the shape in current model is torch.Size([32128, 4096]).
   ```
- Please download the lastest version of `tokenizer.py`:
   ```
   wget https://raw.githubusercontent.com/NVIDIA/Megatron-LM/refs/heads/main/megatron/training/tokenizer/tokenizer.py
   mv tokenizer.py MEGATRON/megatron/training/tokenizer/tokenizer.py
   ```
- This error is because the llama2 and llama3 has different tokenizer design; need to be fixed in the future
