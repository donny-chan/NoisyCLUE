import json
from pathlib import Path

import sacrebleu


def load_jsonl(file):
    return [json.loads(line) for line in open(file, 'r')]

example_file = '../../data/keyboard/nmt/nmt_test_clean.json'
print(f'Loading examples from {example_file}', flush=True)
examples = load_jsonl(example_file)
en_lines = [x['en'] for x in examples][:128]

output_dir = Path('../results/nmt/mbart-large')

for phase in ['clean', 'noisy_1', 'noisy_2', 'noisy_3']:
    print(phase)

    test_dir = output_dir / f'test_{phase}'

    # print('Loading predictions from', test_dir, flush=True)

    preds_file = test_dir / 'preds.json'
    preds = load_jsonl(preds_file)

    # Yes, it is a list of list(s) as required by sacreBLEU
    refs = [en_lines]

    bleu = sacrebleu.corpus_bleu(preds, refs)
    print(bleu.score)
