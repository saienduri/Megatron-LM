HF_FORMAT_DIR=/home/amd/guihong/checkpoint/llama2-7b-hf
MEGATRON_FORMAT_DIR=/home/amd/guihong/checkpoint/llama2-7b-mgm
TOKENIZER_MODEL=/home/amd/guihong/checkpoint/llama2-7b-hf
TP=1


python tools/checkpoint/convert.py \
      --model-type GPT \
      --model-size llama2-7B \
      --loader llama_mistral \
      --saver mcore \
      --target-tensor-parallel-size ${TP} \
      --checkpoint-type hf \
      --load-dir ${HF_FORMAT_DIR} \
      --save-dir ${MEGATRON_FORMAT_DIR} \
      --tokenizer-model ${TOKENIZER_MODEL} \


