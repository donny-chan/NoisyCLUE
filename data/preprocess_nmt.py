from pathlib import Path
from utils import load_jsonl, dump_jsonl


data_dir = Path('keyboard/nmt')
examples = load_jsonl(data_dir / 'nmt_test.json')
test_type_to_key = {
    'clean': 'zh',
    'noisy_1': 'noisy1_zh',
    'noisy_2': 'noisy2_zh',
    'noisy_3': 'noisy3_zh',
}

for test_type, zh_key in test_type_to_key.items():
    parsed_examples = []
    for example in examples:
        parsed_examples.append({
            'id': example['id'],
            'zh': example[zh_key],
            'en': example['en'],
        })
    dump_jsonl(parsed_examples, data_dir / f'nmt_test_{test_type}.json')

