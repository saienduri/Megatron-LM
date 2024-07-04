ORIGIN_MODEL_PATH=/dockerx/Llama-2-7b-hf/
TP=1
OUT_DIR=checkpoints/llama2_7b
mkdir -p $OUT_DIR
RESIZED_MODEL_PATH=$OUT_DIR/hf
MEGATRON_MODEL_PATH=$OUT_DIR/megatron
mkdir -p $RESIZED_MODEL_PATH
mkdir -p $MEGATRON_MODEL_PATH

echo "import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
config = transformers.AutoConfig.from_pretrained(
    '$ORIGIN_MODEL_PATH',
)
tokenizer = transformers.AutoTokenizer.from_pretrained(
    '$ORIGIN_MODEL_PATH',
    padding_side='right',
    use_fast=False,
)
num_tokens = len(tokenizer)

DEFAULT_PAD_TOKEN = '[PAD]'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_BOS_TOKEN = '<s>'
DEFAULT_UNK_TOKEN = '<unk>'
special_tokens_dict = dict()
if tokenizer.pad_token is None:
    special_tokens_dict['pad_token'] = DEFAULT_PAD_TOKEN
if tokenizer.eos_token is None:
    special_tokens_dict['eos_token'] = DEFAULT_EOS_TOKEN
if tokenizer.bos_token is None:
    special_tokens_dict['bos_token'] = DEFAULT_BOS_TOKEN
if tokenizer.unk_token is None:
    special_tokens_dict['unk_token'] = DEFAULT_UNK_TOKEN
special_tokens_dict.update({'additional_special_tokens': ['<extra_id_0>', '<extra_id_1>', '<extra_id_2>']})
tokenizer.add_special_tokens(special_tokens_dict)

model = transformers.AutoModelForCausalLM.from_pretrained(
    '$ORIGIN_MODEL_PATH',
    config=config,
    torch_dtype=torch.bfloat16,
)
config.vocab_size = len(tokenizer)
model.resize_token_embeddings(config.vocab_size, pad_to_multiple_of=$TP)

config.save_pretrained('$RESIZED_MODEL_PATH')
tokenizer.save_pretrained('$RESIZED_MODEL_PATH')
model.save_pretrained('$RESIZED_MODEL_PATH')" > prepare_sft_llama2.py

python prepare_sft_llama2.py

python tools/checkpoint/util.py --model-type GPT --loader llama2_hf --saver megatron --load-dir $RESIZED_MODEL_PATH --save-dir $MEGATRON_MODEL_PATH --tokenizer-model $RESIZED_MODEL_PATH --target-tensor-parallel-size $TP
