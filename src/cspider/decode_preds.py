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
    print(f'Decoding preds in {output_dir}', flush=True)
    file_preds = output_dir / 'preds.json'
    preds = load_jsonl(file_preds)
    preds_text = tokenizer.batch_decode(preds, skip_special_tokens=True)
    file_preds_text = output_dir / 'preds_text.json'
    
    print(f'Dumping to {file_preds_text}', flush=True)
    dump_jsonl(preds_text, file_preds_text)


def decode(tokenizer: MBartTokenizerFast, output_dir: Path):
    # Clean
    try:
        decode_preds(tokenizer, output_dir / 'test_clean')
    except:
        pass
    
    # Noisy
    for noise_type in ['keyboard', 'asr']:
        for i in range(1, 4):
            try:
                decode_preds(tokenizer, output_dir / f'test_noisy_{noise_type}_{i}')
            except:
                pass


def main():
    model_path = 'facebook/mbart-large-cc25'
    tokenizer = MBartTokenizerFast.from_pretrained(model_path)
    lr = "3e-5"
    for lr in [
        '1e-5',
        '2e-5',
        '3e-5',
        '5e-5',
    ]:
        output_dir = Path(f'../results/cspider_norm/mbart-large-cc25_lr{lr}')
        decode(tokenizer, output_dir)


if __name__ == '__main__':
    main()