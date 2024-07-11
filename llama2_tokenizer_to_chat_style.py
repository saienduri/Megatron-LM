import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig



chat_template = "\
{% if not add_generation_prompt is defined %}\
{% set add_generation_prompt = false %}\
{% endif %}\
{{'<extra_id_0>System\n\n'}}\
{% for message in messages %}\
{{'<extra_id_1>'}}{{message['role'] | capitalize}}{{'\n' + message['content'] + '\n'}}\
{% endfor %}\
{% if add_generation_prompt %}\
{{'<extra_id_1>Assistant\n'}}\
{% endif %}"


chat_template_gsm8k = "\
{% if not add_generation_prompt is defined %}\
{% set add_generation_prompt = false %}\
{% endif %}\
{{'<extra_id_0>System\n\n'}}\
{{'Below is a math question. I want you to **first reason through the steps required to reach the answer**, then end your **response with \\\"#### \\\" followed by the answer**. For instance, if the answer is 42 then your response must end with \\\"#### 42\\\" (without the quotes).\n'}}\
{% for message in messages %}\
{{'<extra_id_1>'}}{{message['role'] | capitalize}}{{'\n' + message['content'] + '\n'}}\
{% endfor %}\
{% if add_generation_prompt %}\
{{'<extra_id_1>Assistant\n'}}\
{% endif %}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-path", type=str, default="checkpoints/llama2_7b/hf_chat", help="Base directory of HF tokenizer repository")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.hf_path, use_fast=False)

    tokenizer.chat_template = chat_template

    turn_start_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('<extra_id_1>'))[0]
    eos_token_id = tokenizer.eos_token_id
    tokenizer.pad_token_id = eos_token_id
    tokenizer.eos_token_id = turn_start_tokens
    tokenizer.save_pretrained(args.hf_path)


if __name__ == "__main__":
    main()

