from pathlib import Path
import json
from utils import load_jsonl


def preprocess(data_dir):
    examples = load_jsonl(data_dir / 'test.json')
    for test_type in ['clean', 'noisy_1', 'noisy_2', 'noisy_3']:
        dst_file = data_dir / f'test_{test_type}.json'
        with open(dst_file, 'w') as f:
            for example in examples:
                pref = test_type.replace('_', '')
                question = example[f'{pref}_question']
                example = {
                    'id': example['id'],
                    'context': example['context'],
                    'question': question,
                    'answers': example['answers'],
                }
                f.write(json.dumps(example, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    data_dir = Path('keyboard/cmrc2018')
    preprocess(data_dir)
