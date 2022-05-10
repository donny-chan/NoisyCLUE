from pathlib import Path

from utils import load_jsonl, dump_jsonl


test_file = Path('keyboard/cluener/ner_test.json')
raw_examples = load_jsonl(test_file)

test_types = {
    'clean': 'clean_text',
    'noisy_1': 'text1',
    'noisy_2': 'text2',
    'noisy_3': 'text3',
}

for test_type, key in test_types.items():
    examples = [{
        'id': x['id'],
        'text': x[key],
        'label': x['label']
    } for x in raw_examples]
    dst_file = test_file.parent / f'ner_test_{test_type}.json'
    print(f'Dumping to {dst_file}')
    
    dump_jsonl(examples, dst_file)