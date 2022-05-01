import json

def load_tsv(file):
    lines = open(file, 'r').readlines() 
    lines = [line.strip().split('\t') for line in lines]
    return lines

def dump_json(data, file):
    with open(file, 'w', encoding='utf8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')


def parse_noisy(lines, speaker_idx):
    data = []
    s1_col = 2 + speaker_idx * 2
    s2_col = s1_col + 1
    for line in lines[1:]:
        data.append({
            'sentence1': line[s1_col],
            'sentence2': line[s2_col],
            'label': line[1],
        })
    return data

def to_dicts(lines):
    return[{
        'sentence1': line[2],
        'sentence2': line[3],
        'label': line[1],
    } for line in lines]

def process_train_data(data_dir: str):
    for fname in ['train', 'dev']:
        lines = load_tsv(f'{data_dir}/afqmc_{fname}.tsv')[1:]
        data = to_dicts(lines)
        dump_json(data, f'{data_dir}/{fname}.json')


def process_test_data(data_dir: str):
    noisy_data = load_tsv(f'{data_dir}/afqmc_test.tsv')
    for i, name in enumerate(['clean', 'noisy_1', 'noisy_2', 'noisy_3']):
        parsed = parse_noisy(noisy_data, i)
        dump_json(parsed, f'{data_dir}/test_{name}.json')


data_dir = 'afqmc_unbalanced'
print('processing train data...')
process_train_data(data_dir)
print('processing test data...')
process_test_data(data_dir)
print('done')
