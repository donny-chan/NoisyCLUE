'''
Tokenize all documents.
'''
from time import time
from pathlib import Path
import json


def get_tokenizer():
    from transformers import AutoTokenizer
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained('google/mt5-base')
    return tokenizer


data_dir = Path('../../data/un-parallel/en-zh')
zh_file = data_dir / 'UNv1.0.en-zh.zh'
en_file = data_dir / 'UNv1.0.en-zh.en'


def iter_lines(file, chunk_size=128*1024):
    with open(file, 'r') as f:
        chunk = []
        for line in f:
            chunk.append(line.strip())
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk != []:
            yield chunk

print('Getting tokenizer...', flush=True)
tokenizer = get_tokenizer()

print('*** Start generating features ***', flush=True)
dst_file = 'features.json'
writer = open(dst_file, 'w', encoding='utf8')
processed_cnt = 0
total = 15886041
start_time = time()
for en_lines, zh_lines in zip(iter_lines(en_file), iter_lines(zh_file)):
    kwargs = {
        'max_length': 128,
        'padding': 'max_length',
        'truncation': True,
    }
    zh = tokenizer(zh_lines, **kwargs)
    en = tokenizer(en_lines, **kwargs)

    count = len(zh['input_ids'])
    for i in range(count):
        feat = {
            'input_ids': zh['input_ids'][i],
            'attention_mask': zh['attention_mask'][i],
            'labels': en['input_ids'][i],
        }
        writer.write(json.dumps(feat) + '\n')
    processed_cnt += count
    time_elapsed = time() - start_time
    processed_prop = processed_cnt / total
    print(f'Processed [{processed_cnt}/{total}], time_elapsed: {time_elapsed}, expected_time: {time_elapsed / processed_prop}', flush=True)

print('done')
