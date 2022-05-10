from pathlib import Path
import json
from utils import load_jsonl

data_dir = Path('keyboard/cmrc2018')
examples = load_jsonl(data_dir / 'cmrc2018_test.json')

for test_type in ['clean', 'noisy_1', 'noisy_2', 'noisy_3']:
    dst_file = data_dir / f'cmrc2018_test_{test_type}.json'
    print(f'Saving to {dst_file}')
    with open(dst_file, 'w') as f:
        for example in examples:
            question = example[f'{test_type}_question']
            example = {
                'id': example['id'],
                'context': example['context'],
                'question': question,
                'answers': example['answers'],
            }
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

