import requests
from starlette.requests import Request
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from typing import Dict

from transformers import pipeline

from ray import serve


# 1: Wrap the pretrained sentiment analysis model in a Serve deployment.
@serve.deployment(ray_actor_options={"num_gpus": 1})
class SentimentAnalysisDeployment:
    def __init__(self, seq_length=2048):
        self.tokenizer = AutoTokenizer.from_pretrained("checkpoints/llama2_7b/hf_chat", use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained("checkpoints/llama2_7b/hf_chat").cuda()
        turn_start_tokens = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id

        self.generation_config = GenerationConfig(
            max_length=seq_length,
            max_new_tokens=512,
            # max_new_tokens=None,
            do_sample=False, 
            top_k=50, 
            top_p=0.95,
            # eos_token_id=model.config.eos_token_id,
            eos_token_id=turn_start_tokens,
            pad_token_id=pad_token_id,
            use_cache=True,
            # repetition_penalty=1.5,
        )

    def __call__(self, request: Request) -> Dict:
        text = request.query_params["text"]
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs['input_ids'] = inputs['input_ids'][:, 1:].cuda()
        inputs['attention_mask'] = inputs['attention_mask'][:, 1:].cuda()
        outputs = self.model.generate(generation_config=self.generation_config, **inputs) 
        res = self.tokenizer.decode(outputs[0])
        return {"res": res}


# 2: Deploy the deployment.
serve.run(SentimentAnalysisDeployment.bind(), route_prefix="/")

# # 3: Query the deployment and print the result.
# print(
#     requests.get(
#         "http://localhost:8000/", params={"text": "Ray Serve is great!"}
#     ).json()
# )
# # {'label': 'POSITIVE', 'score': 0.9998476505279541}
