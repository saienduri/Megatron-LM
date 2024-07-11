```bash
# Tune the training GEMM config
bash tune_and_train_llama2.sh

# Prepare the megatron weights
bash prepare_sft_llama2.sh

# Run training
bash sft_llama2.sh

# Convert megatron weights to HF
SFT_MEGATRON_MODEL_PATH=checkpoints/llama2_7b/megatron_chat_4k_openhermes_2_5_lr3e-7_bs128/iter_0001600/ bash llama2_to_hf.sh
```