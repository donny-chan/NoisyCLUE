from pathlib import Path

# from torchtext.data.metrics import bleu_score
from nltk.translate.bleu_score import corpus_bleu

from utils import load_json, load_jsonl, dump_jsonl

def get_tokenizer():
    print('Importing transformers', flush=True)
    from transformers import AutoTokenizer
    print('Loading tokenizer...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained('google/mt5-base')
    return tokenizer

tokenizer = get_tokenizer()

output_dir = Path('results/un_parallel/mt5-small_const-lr1e-4')
data_dir = Path('../data/keyboard/nmt')

for test_phase in ['clean', 'noisy_1', 'noisy_2', 'noisy_3']:
    print(f'{test_phase}')
    labels_file = data_dir / f'nmt_test_{test_phase}.json'
    labels = load_jsonl(labels_file)
    cands = [[x['en'].split(' ')] for x in labels]

    preds_file = output_dir / f'test_{test_phase}' / 'pred_texts.json'
    preds = load_jsonl(preds_file)
    refs = [p.split(' ') for p in preds]
    dump_jsonl(cands, 'cands.json')
    dump_jsonl(refs, 'refs.json')
    bleu = corpus_bleu(cands, refs)
    print(f'BLEU-4: {bleu}')

