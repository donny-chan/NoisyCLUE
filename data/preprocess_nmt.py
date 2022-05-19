from pathlib import Path
from utils import load_jsonl, dump_jsonl


def preprocess(data_dir):
    examples = load_jsonl(data_dir / 'test.json')
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
        dump_jsonl(parsed_examples, data_dir / f'test_{test_type}.json')

if __name__ == '__main__':
    data_dir = Path('asr/nmt')
    preprocess(data_dir)
