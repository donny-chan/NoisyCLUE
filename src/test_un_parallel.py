import torch
assert torch.cuda.is_available()

from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import dump_json, dump_jsonl
from un_parallel.trainer import UNParallelTrainer
from un_parallel.data import UNParallelZhEnIterableDataset

results_dir = Path('results/un_parallel')

path = 'mt5-base'
# output_dir = results_dir / f'{path}_const-lr1e-4'
# ckpt_dir = output_dir / 'checkpoint-13312'

# output_dir = results_dir / f'{path}_lr1e-4'
# ckpt_dir = output_dir / 'checkpoint-6144'

# path = 'mt5-small'
# output_dir = results_dir / f'{path}_const-lr1e-4'
# ckpt_dir = output_dir / 'checkpoint-5120'

# model_file = ckpt_dir / 'pytorch_model.bin'
# path = f'google/{path}'
path = 'K024/mt5-zh-ja-en-trimmed'
output_dir = results_dir / 'temp-K024'

path = 'Helsinki-NLP/opus-mt-zh-en'
output_dir = results_dir / 'temp-opus'

def test_all(trainer, tokenizer):
    for test_phase in ['clean', 'noisy_1', 'noisy_2', 'noisy_3']:
        features_file = f'un_parallel/test_features_{test_phase}.json'
        print(f'Building data from {features_file}')
        dataset = UNParallelZhEnIterableDataset(features_file, cache_size=2**10, num_examples=200)
        dataloader = DataLoader(dataset, batch_size=8)
        output_dir = trainer.output_dir / f'test_{test_phase}'
        output_dir.mkdir(exist_ok=True)


        print(f'Testing {test_phase}', flush=True)
        # model = trainer.model.cuda()
        model.eval()
        
        pred_texts = []
        all_label_texts = []
        preds = []

        for step, batch in tqdm(enumerate(dataloader), total=25):
            output_seq = model.generate(
                input_ids=batch['input_ids'].cuda(),
                attention_mask=batch['attention_mask'].cuda(),
                max_length=128,
                do_sample=False,
            )
            output_texts = tokenizer.batch_decode(output_seq, skip_special_tokens=True)
            label_texts =tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
            preds += output_seq.tolist()
            pred_texts += output_texts
            all_label_texts += label_texts

        # eval_output = trainer.evaluate(dataset, output_dir, test_phase)
        
        print(f'Dumping {test_phase} test result to {output_dir}')
        # result = eval_output['result']
        # preds = eval_output['preds']
        # dump_json(result, output_dir / f'result.json')
        dump_jsonl(preds, output_dir / f'preds.json')
        # pred_texts = [tokenizer.decode(x, skip_special_tokens=True) for x in preds]
        dump_jsonl(pred_texts, output_dir / f'pred_texts.json')
        dump_jsonl(label_texts, output_dir / f'label_texts.json')

def get_trainer(model, output_dir):
    trainer = UNParallelTrainer(
        model,
        output_dir,
        batch_size=8,
        log_interval=1,
        log_file='test.log',
    )
    return trainer

output_dir.mkdir(exist_ok=True, parents=True)
print(f'output dir: {output_dir}')
print('Loading model...', flush=True)
model = AutoModelForSeq2SeqLM.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)
print('Setting up trainer...', flush=True)
trainer = get_trainer(model, output_dir)
# print(f'Loading from {model_file}', flush=True)
# trainer.load_model(model_file)

test_all(trainer, tokenizer)
