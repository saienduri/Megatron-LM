# import json
# fin = open("/dockerx/wizard_platypus_sharegpt4/train_llama_s2048.jsonl", 'r', encoding='utf-8')
# for jsonl in fin:
#     # import pudb; pu.db
#     data = json.loads(jsonl)
#     break

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("checkpoints/llama2_7b/hf_chat")
model = AutoModelForCausalLM.from_pretrained("checkpoints/llama2_7b/hf_chat")

messages = [
    {"role": "User", "content": "How many helicopters can a human eat in one sitting?"},
 ]


tokenizer.chat_template = """
{% if not add_generation_prompt is defined %}
{% set add_generation_prompt = false %}
{% endif %}
{{ '<extra_id_0>System' + '\n' + 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\\'s questions.\n\n' + '\n' }}
{% for message in messages %}
{{'<extra_id_1>' + message['role'] + '\n' + message['content'] + '\n'}}
{% endfor %}
{% if add_generation_prompt %}
{{ '<extra_id_1>Assistant\n' }}
{% endif %}
"""

# import pudb; pu.db
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
tokenized_chat_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
print(tokenized_chat)

outputs = model.generate(tokenized_chat_ids, max_new_tokens=128) 
print(tokenizer.decode(outputs[0]))