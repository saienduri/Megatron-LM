# How to run
## Environment Setup

### Docker and library
Pull the `rocm/pytorch-private:exec_dash_pretuned_nightly_inai_FA_ck_v0.1.1_TE` docker image. 

Example:
<pre>docker run -it --device /dev/dri --device /dev/kfd --network host --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged    -v  $HOME/.ssh:/root/.ssh  --shm-size 128G --name llama-70b-training-gl  rocm/pytorch-private:exec_dash_pretuned_nightly_inai_FA_ck_v0.1.1_TE
</pre>

Install the afo-based gemm tuning librabry
```
cd pytorch_afo_testkit
pip install -e . && cd ..
```


<!-- ### On H100
Pull the `nvcr.io/nvidia/pytorch:24.07-py3` docker image.

Install the following dependencies:
<pre>
pip install ftfy datasets langdetect flash_attn numpy pandas nltk sentencepiece boto3 tqdm regex bs4 newspaper3k htmlmin tldextract transformers

pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
</pre> -->

### Network configuration
Before run the training, we need to adapt the network interface.
- Network interface
   - Currently, we are using network interface `ens51f0np0` in lines-{11,12} at our script [train70b_acc_loss.sh](./train70b_acc_loss.sh).
   - We can reset `ens51f0np0` to the network interface of the used server; `ifconfig` can list all available network interfaces.
- Infini-Band
   - For single-node test, we don't need Infini-Band, please comment out lines-{8,9}; 
   - For multi-node training, we need to configure the `NCCL_IB_HCA` and `NCCL_IB_GID_INDEX` env vars in 

### Prepare the dataset:
Install `git lfs` and process the dataset file:
```
git clone https://huggingface.co/datasets/teknium/OpenHermes-2.5 PATH_TO_DATASET
python openhermes_2_5_to_jsonl.py --data_file=$PATH_TO_DATASET/openhermes2_5.json
```
After processing, there will be a 'openhermes2_5.jsonl' file in PATH_TO_DATASET.

Modify the data path to `$PATH_TO_DATASET/openhermes2_5.jsonl` in lines-{68,69} at our script [train70b_acc_loss.sh](./train70b_acc_loss.sh).

### Prepare the tokenizer:
Download the correpsonding tokenizer from huggingface. 

For example, for llama2-70b model, download the `tokenizer.model`, `config.json` and `tokenizer_config.json` from [Hugging-face](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf/tree/main)

Assume the tokenizer is stored at TOKENIZER_PATH

Modify the tokenizer path to `$TOKENIZER_PATH` in lines-70 at our script [train70b_acc_loss.sh](./train70b_acc_loss.sh).

## Running the benchmarking


### Single node with performance tuning
Set the parameters in [tune_basetrain.sh](./tune_basetrain.sh) or in the bash command as follows:
<pre>
bash tune_basetrain.sh NO_TORCH_COMPILE=0 BASHFILE=train70b_acc_loss.sh MODEL_SIZE=70 MBS=4 BS=32 NO_TORCH_COMPILE=0
</pre>

### Multiple node
Download this repo to each of the node

Run exactly the same setup (docker, python-environment) in every node

Set the parameters in [train70b_acc_loss.sh](./train70b_acc_loss.sh)

For each node, modify the following lines in [train70b_acc_loss.sh](./train70b_acc_loss.sh)
<pre>
export GLOO_SOCKET_IFNAME=ens51f0np0 --> to the network interface on the server [can by obtainted by run ifconfig]
export NCCL_SOCKET_IFNAME=ens51f0np0 --> to the network interface on the server [can by obtainted by run ifconfig]
MASTER_ADDR=localhost --> to the IP address of the master node (rank=0)
MASTER_PORT=23731 --> use a free port number
NNODES=1 --> to the number of nodes
NODE_RANK=0 --> to the rank for this server; this is NODE-dependent!
</pre>



On each node:
<pre>
bash tune_basetrain.sh NO_TORCH_COMPILE=0
</pre>

