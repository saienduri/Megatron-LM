import json
from pathlib import Path
from tqdm import tqdm
FILE = "/dockerx/alpaca-lora/alpaca_data_cleaned_archive.json"
OUT_FILE = Path(FILE).with_suffix(".jsonl")
with open(FILE, "r") as f:
    data = json.load(f)

with open(OUT_FILE, "w") as f:
    for i, d in tqdm(enumerate(data)):
        if d["input"]:
            input_str = f'{d["instruction"]} {d["input"]}'
        else:
            input_str = f'{d["instruction"]}{d["input"]}'
        data = {
            "mask": "User",
            "system": "",
            "conversations": [
                {"from": "User", "value": input_str},
                {"from": "Assistant", "value": d["output"]}
            ]
        }
        f.write(json.dumps(data) + "\n")
        # if i > 10:
            # break
