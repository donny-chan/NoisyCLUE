import json

def load_jsonl(file):
    return [json.loads(line) for line in open(file, 'r')]

def iter_jsonl(file):
    for line in open(file, 'r'):
        yield json.loads(line)

def dump_jsonl(data, file):
    # print('Dumping to:', file)
    with open(file, 'w', encoding='utf8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
            
def load_tsv(file):
    lines = open(file, 'r').readlines() 
    lines = [line.split('\t') for line in lines]
    return lines

def load_json(file):
    return json.load(open(file, 'r'))

def dump_json(data, file, **kwargs):
    json.dump(data, open(file, 'w', encoding='utf8'), ensure_ascii=False, **kwargs)