from utils import dump_jsonl, load_tsv

def parse_noisy(lines, speaker_idx):
    data = []
    s1_col = 2 + speaker_idx * 2
    s2_col = s1_col + 1
    for line in lines[1:]:
        # print(line)
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
        dump_jsonl(data, f'{data_dir}/{fname}.json')

def process_test_data(data_dir: str):
    noisy_data = load_tsv(f'{data_dir}/afqmc_test.tsv')
    for i, name in enumerate(['clean', 'noisy_1', 'noisy_2', 'noisy_3']):
        parsed = parse_noisy(noisy_data, i)
        dump_jsonl(parsed, f'{data_dir}/test_{name}.json')

def preprocess(data_dir):
    # print('Processing', data_dir)
    print('processing train data...')
    process_train_data(data_dir)
    print('processing test data...')
    process_test_data(data_dir)
    print('done')

if __name__ == '__main__':
    DATA_DIR = 'keyboard'
    for data_dir in [
        'afqmc_unbalanced', 
        'afqmc_balanced'
        ]:
        data_dir = f'{DATA_DIR}/{data_dir}'
        preprocess(data_dir)
