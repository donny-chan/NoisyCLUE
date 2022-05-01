import json
import torch

def iter_jsonl(file):
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            yield json.loads(line.strip())
