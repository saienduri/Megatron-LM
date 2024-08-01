import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
config = transformers.AutoConfig.from_pretrained(
    '../../huggingface/llama2-70b',
)
tokenizer = transformers.AutoTokenizer.from_pretrained(
    '../../huggingface/llama2-70b',
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
    '../../huggingface/llama2-70b',
    config=config,
    torch_dtype=torch.bfloat16,
)
config.vocab_size = len(tokenizer)
model.resize_token_embeddings(config.vocab_size, pad_to_multiple_of=8)

config.save_pretrained('checkpoints/llama2_70b/hf')
tokenizer.save_pretrained('checkpoints/llama2_70b/hf')
model.save_pretrained('checkpoints/llama2_70b/hf')
