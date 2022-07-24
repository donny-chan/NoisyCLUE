from pathlib import Path
import json

from transformers import MBartTokenizerFast


def load_jsonl(file):
    return [json.loads(line) for line in open(file, 'r')]


def dump_jsonl(data, file):
    with open(file, 'w', encoding='utf8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
        


def decode_preds(tokenizer, output_dir):
    '''
    Decode "preds.json" and output to "preds_text.json"
    '''
    print(f'Loading preds from {output_dir}', flush=True)
    file_preds = output_dir / 'preds.json'
    preds = json.load(open(file_preds, 'r'))  # dev preds are dumping as raw JSON
    print(f'Decoding preds', flush=True)
    preds_text = tokenizer.batch_decode(preds, skip_special_tokens=True)
    file_preds_text = output_dir / 'preds_text.json'
    
    print(f'Dumping to {file_preds_text}', flush=True)
    dump_jsonl(preds_text, file_preds_text)


def decode(tokenizer: MBartTokenizerFast, output_dir: Path):
    # Clean
    decode_preds(tokenizer, output_dir / 'test_clean')
    
    # Noisy
    for noise_type in ['keyboard', 'asr']:
        for i in range(1, 4):
            decode_preds(tokenizer, output_dir / f'test_noisy_{noise_type}_{i}')


def main():
    model_path = 'facebook/mbart-large-cc25'
    tokenizer = MBartTokenizerFast.from_pretrained(model_path)
    lr = "2e-4"
    output_dir = Path(f'../results/cspider_flat_schema/mbart-large-cc25_lr{lr}')
    
    decode_preds(tokenizer, output_dir / 'checkpoint-4')


if __name__ == '__main__':
    main()