from pathlib import Path
import json

output_file = 'afqmc/split/noisy_test.json'
clean_file = 'afqmc/dev.json'
src_dir = 'all_tsv'


def load_tsv(file):
    lines = open(file, 'r', encoding='utf-8-sig').readlines()
    lines = [line.strip().split(',') for line in lines]
    data = {}
    for i in range(0, len(lines), 2):
        name = lines[i][0]
        idx = name.split('_')[2]
        data[int(idx)] = {
            'sentence1': lines[i][1],
            'sentence2': lines[i+1][1]
        }
    return data

def load_noisy(dir):
    all_data = {}
    for file in Path(dir).glob('afqmc_*'):
        data = load_tsv(file)
        all_data.update(data)
    return [all_data[k] for k in sorted(all_data)]

def load_jsonl(file):
    data = []
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def dump_jsonl(data, file):
    with open(file, 'w', encoding='utf8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')


noisy = load_noisy(src_dir)
clean = load_jsonl(clean_file)
for i in range(len(noisy)):
    noisy[i]['label'] = clean[i]['label']

dump_jsonl(noisy, output_file)