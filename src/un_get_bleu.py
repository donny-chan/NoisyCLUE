from pathlib import Path

from utils import load_json, load_jsonl

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
    labels_file = data_dir / f'nmt_test_{test_phase}.json'
    labels = load_jsonl(labels_file)
    refs = [x['en'] for x in labels]
    preds_file = output_dir / f'test_{test_phase}' / 'preds.json'
    preds = load_jsonl(preds_file)[0]
    cands = []
    for pred in preds:
        print(pred)
        cand = tokenizer.decode(pred)
        print(cand)
        exit()
    preds = tokenizer.decode(preds)
    print(preds[:10])
    exit()


