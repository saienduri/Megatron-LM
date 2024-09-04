import json
from pathlib import Path
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='scan_multi_nodes',
                                    allow_abbrev=False)
parser.add_argument('--data_file', type=str, default=None, 
                    help='the path of data file')
args = parser.parse_args()
FILE = args.data_file

OUT_FILE = Path(FILE).with_suffix(".jsonl")
with open(FILE, "r") as f:
    data = json.load(f)
print(f"Num samples: {len(data)}")

with open(OUT_FILE, "w") as f:
    for i, d in enumerate(tqdm(data)):
        conversations = d['conversations']
        system = ''
        if len(conversations) == 3:
            assert conversations[0]['from'] == 'system'
            system = conversations[0]['value']
            conversations = conversations[1:]
        assert len(conversations) % 2 == 0
        converted_conversations = []
        for human, gpt in zip(conversations[::2], conversations[1::2]):
            assert human['from'] == 'human'
            assert gpt['from'] == 'gpt'
            input_str = human['value']
            label_str = gpt['value']
            converted_conversations.extend([
                {"from": "User", "value": input_str},
                {"from": "Assistant", "value": label_str}
            ])
        data = {
            "mask": "User",
            "system": system,
            "conversations": converted_conversations
        }
        f.write(json.dumps(data) + "\n")
