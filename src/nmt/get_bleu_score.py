import json
from pathlib import Path

import sacrebleu


def load_jsonl(file):
    return [json.loads(line) for line in open(file, 'r')]

example_file = '../../data/keyboard/nmt/nmt_test_clean.json'
examples = load_jsonl(example_file)
en_lines = [x['en'] for x in examples]

output_dir = Path('../results/nmt/mbart-large')

scores = []
phases = ['clean', 'noisy_1', 'noisy_2', 'noisy_3']
for phase in phases:
    print(f'---- {phase} ----')
    test_dir = output_dir / f'test_{phase}'
    preds = load_jsonl(test_dir / 'preds.json')

    refs = [en_lines]  # Yes, it is a list of list(s) as required by sacreBLEU

    bleu = sacrebleu.corpus_bleu(preds, refs)
    score = bleu.score
    scores.append(score)
    print(bleu)
    print(f'BLEU: {score}')

    json.dump(vars(bleu), open(test_dir / 'bleu.json', 'w'), indent=2)

print('---- Average ----')
print(sum(scores) / len(scores))
