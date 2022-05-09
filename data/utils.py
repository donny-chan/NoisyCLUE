import json

def load_jsonl(file):
    return [json.loads(line) for line in open(file, 'r')]

def iter_jsonl(file):
    for line in open(file, 'r'):
        yield json.loads(line)

def dump_jsonl(data, file):
    with open(file, 'w', encoding='utf8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')