OUT_DIR=checkpoints/llama2_7b
HF_MODEL_PATH=$OUT_DIR/hf
SFT_MEGATRON_MODEL_PATH="${SFT_MEGATRON_MODEL_PATH:-$OUT_DIR/megatron_chat}"
SFT_HF_MODEL_PATH=$OUT_DIR/hf_chat
rm -rf $SFT_HF_MODEL_PATH
mkdir -p $SFT_HF_MODEL_PATH
python llama2_megatron_to_hf.py --load_path $SFT_MEGATRON_MODEL_PATH --save_path $SFT_HF_MODEL_PATH --target_params_dtype "fp16" --make_vocab_size_divisible_by 1 --print-checkpoint-structure --megatron-path .

cp -R $HF_MODEL_PATH/token* $SFT_HF_MODEL_PATH
