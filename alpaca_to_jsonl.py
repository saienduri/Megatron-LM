import json
from pathlib import Path
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser(description="data conversion")
parser.add_argument('--input_file', type=str, required=True,
                    help='input json file')
parser.add_argument('--output_file', type=str, default=None,
                    help='input json file')
args, _ = parser.parse_known_args()

if args.output_file is None:
    args.output_file = Path(args.input_file).with_suffix(".jsonl")

with open(args.input_file, "r") as f:
    data = json.load(f)

with open(args.output_file, "w") as f:
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
