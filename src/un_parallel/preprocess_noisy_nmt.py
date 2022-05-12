'''
Tokenize all documents.
'''
from time import time
from pathlib import Path
import json


def load_examples(file):
    for line in open(file, 'r'):
        yield json.loads(line)


def get_tokenizer():
    print('Importing transformers', flush=True)
    from transformers import AutoTokenizer
    print('Loading tokenizer...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained('google/mt5-base')
    return tokenizer


def get_features(file, tokenizer):
    examples = list(load_examples(file))
    en_lines = [x['en'] for x in examples]
    zh_lines = [x['zh'] for x in examples]

    kwargs = {
        'max_length': 128,
        'padding': 'max_length',
        'truncation': True,
    }
    inputs = tokenizer(zh_lines, **kwargs)
    labels = tokenizer(en_lines, **kwargs)

    features = []
    count = len(inputs['input_ids'])
    for i in range(count):
        feat = {
            'input_ids': inputs['input_ids'][i],
            'attention_mask': inputs['attention_mask'][i],
            'labels': labels['input_ids'][i],
        }
        features.append(feat)
    return features


tokenizer = get_tokenizer()
print('*** Start generating features ***', flush=True)
dst_file = 'test_features.json'

data_dir = Path('../../data/keyboard/nmt')
for suffix in ['clean', 'noisy_1', 'noisy_2', 'noisy_3']:
    file = data_dir / f'nmt_test_{suffix}.json'
    print('getting from:', file)
    features = get_features(file, tokenizer)

    dst_file = f'test_features_{suffix}.json'
    print('writing to:', dst_file)
    with open(dst_file, 'w', encoding='utf8') as f:
        for feat in features:
            f.write(json.dumps(feat) + '\n')
