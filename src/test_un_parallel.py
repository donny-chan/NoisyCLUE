import torch
assert torch.cuda.is_available()

from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from utils import dump_json, dump_jsonl
from un_parallel.trainer import UNParallelTrainer
from un_parallel.data import UNParallelZhEnIterableDataset


output_dir = Path('results/un_parallel/mt5-small_const-lr1e-4')
ckpt_dir = output_dir / 'checkpoint-1'
model_file = ckpt_dir / 'pytorch_model.bin'


def test_all(trainer, tokenizer):
    for test_phase in ['clean', 'noisy_1', 'noisy_2', 'noisy_3']:
        print(f'Testing {test_phase}', flush=True)
        features_file = f'un_parallel/test_features_{test_phase}.json'
        print(f'Building data from {features_file}')
        dataset = UNParallelZhEnIterableDataset(features_file, cache_size=2**10, num_examples=1948)
        output_dir = trainer.output_dir / f'test_{test_phase}'

        eval_output = trainer.evaluate(dataset, output_dir, test_phase)
        
        print(f'Dumping {test_phase} test result to {output_dir}')
        result = eval_output['result']
        preds = eval_output['preds'][0]
        dump_json(result, output_dir / f'result.json')
        dump_jsonl(preds, output_dir / f'preds.json')
        pred_texts = [tokenizer.decode(x) for x in preds]
        dump_jsonl(pred_texts, output_dir / f'pred_texts.json')

def get_trainer(model, output_dir):
    trainer = UNParallelTrainer(
        model,
        output_dir,
        batch_size=2,
        log_interval=1,
        log_file='test.log',
    )
    return trainer


print('Loading model...', flush=True)
path = 'google/mt5-base'
model = AutoModelForSeq2SeqLM.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)
print('Setting up trainer...', flush=True)
trainer = get_trainer(model, output_dir)
print(f'Loading from {model_file}', flush=True)
trainer.load_model(model_file)

test_all(trainer, tokenizer)
