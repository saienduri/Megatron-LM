1. Prepare the megatron weights, modify ORIGIN_MODEL_PATH (pretrained weights) in prepare_sft_llama2.sh if necessary 

    ```bash
    bash prepare_sft_llama2.sh
    ```

2. Prepare the dataset

    ```bash
    git clone https://huggingface.co/datasets/teknium/OpenHermes-2.5 /dockerx/OpenHermes-2.5
    python openhermes_2_5_to_jsonl.py
    ```

3. Run training, modify `NICs/data path/weight path/master ip/master port` in `finetune_llama2_chat.sh` if necessary

    ```bash
    bash sft_llama2.sh
    ```

4. Convert megatron weights to HF, modify the following model path is necessary

    ```bash
    SFT_MEGATRON_MODEL_PATH=checkpoints/llama2_70b/megatron_chat_4k_openhermes_2_5_lr1e-5_bs128/iter_0001600/ bash llama2_to_hf.sh
    ```

5. change https://github.com/ROCm/Megatron-LM/blob/82ba2f85c0f750ff1db06d314c62bb20d019fba6/llama2_tokenizer_to_chat_style.py#L41

    ```python
    tokenizer.chat_template = chat_template
    ```

    to

    ```python
    tokenizer.chat_template = chat_template_gsm8k
    ```

6. evaluate using lm_eval

    ```bash
    accelerate launch -m lm_eval --model hf --model_args pretrained=checkpoints/llama2_70b/hf_chat --tasks gsm8k --device cuda --batch_size 8
    ```
