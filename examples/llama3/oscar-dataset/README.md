---
task_categories:
- text-generation
language:
- en
size_categories:
- 1K<n<10K
---
# README
This dataset is used to test the training of the LLAMA model with Megatron-Deepspeed.

## Source
stas/oscar-en-10k

## Command
```bash
python tools/preprocess_data.py \
  --input oscar-en-10k.jsonl \
  --tokenizer-type PretrainedFromHF \
  --append-eod \
  --tokenizer-name-or-path hf-internal-testing/llama-tokenizer \
  --output-prefix oscar-en-10k
```

## Info
\#tokens = 34030076