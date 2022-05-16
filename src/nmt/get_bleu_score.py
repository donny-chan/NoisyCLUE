import json
from pathlib import Path
import logging

from sacrebleu import corpus_bleu, sentence_bleu

logging.getLogger('sacrebleu').setLevel(logging.ERROR)

def avg(x): return sum(x) / len(x)

def load_jsonl(file):
    return [json.loads(line) for line in open(file, 'r')]

example_file = '../../data/keyboard/nmt/nmt_test_clean.json'
examples = load_jsonl(example_file)
en_lines = [x['en'] for x in examples]

output_dir = Path('../results/nmt/mbart-large')

scores = []
all_scores = []  # List of list of sentence BLEU scores
phases = ['clean', 'noisy_1', 'noisy_2', 'noisy_3']
for phase in phases:
    print(f'---- {phase} ----')
    test_dir = output_dir / f'test_{phase}'

    num = None

    hyps = load_jsonl(test_dir / 'preds.json')[:num]
    refs = [en_lines[:num]]  # Yes, it is a list of list(s) as required by sacreBLEU

    bleu = corpus_bleu(hyps, refs)
    score = bleu.score
    scores.append(score)
    print(bleu)
    # json.dump(vars(bleu), open(test_dir / 'bleu.json', 'w'), indent=2)
    
    # hyps = ["In recent years, the number of wildlife species has continued to increase as the ecological environment continues to improve, the city's forestry and greening bureau was informed by a reporter for the Beijing Youth Daily (Wang Bin)."]
    # refs = ["In recent years, as the capital's natural ecosystem continues to improve, the number and variety of wildlife has been growing steadily, the Beijing Youth Daily learned from the City Gardens and Greening Bureau"]
    # refs = [refs]
    # print(corpus_bleu(hyps, refs))
    # print(sentence_bleu(hyps[0], refs[0]))
    # exit()

    print('Micro average')
    sent_scores = []
    for hyp, ref in zip(hyps, refs[0]):
        bleu = sentence_bleu(hyp, [ref])
        # bleu = corpus_bleu([hyp], [[ref]])
        sent_scores.append(bleu.score)
    print(f'BLEU: {avg(sent_scores)}')
    all_scores.append(sent_scores)
    json.dump(sent_scores, open(test_dir / 'bleu_sent.json', 'w'), indent=2)


print('---- Average ----')
print(sum(scores[1:]) / 3)
print('---- Worst group BLEU ----')
worst_scores = []
for sent_scores in zip(*all_scores):
    worst_scores.append(min(sent_scores))
print(sum(worst_scores) / len(worst_scores))


