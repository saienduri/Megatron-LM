```bash
# Tune the training GEMM config
bash tune_and_train_llama2.sh

# Prepare the megatron weights
bash prepare_sft_llama2.sh

# Prepare the dataset
git clone https://huggingface.co/datasets/teknium/OpenHermes-2.5 /dockerx/OpenHermes-2.5
python openhermes_2_5_to_jsonl.py

# Run training
bash sft_llama2.sh

# Convert megatron weights to HF
SFT_MEGATRON_MODEL_PATH=checkpoints/llama2_7b/megatron_chat_4k_openhermes_2_5_lr3e-7_bs128/iter_0001600/ bash llama2_to_hf.sh

# For testing on GSM8K
# change https://github.com/ROCm/Megatron-LM/blob/82ba2f85c0f750ff1db06d314c62bb20d019fba6/llama2_tokenizer_to_chat_style.py#L41
# From tokenizer.chat_template = chat_template to tokenizer.chat_template = chat_template_gsm8k
```
