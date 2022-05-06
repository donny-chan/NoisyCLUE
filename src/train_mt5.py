import torch
assert torch.cuda.is_available()

import time
from pathlib import Path
from argparse import Namespace

from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from data.afqmc import AfqmcSeq2SeqDataset
import utils
import utils_seq2seq
from arguments import parse_args


def train(trainer: Seq2SeqTrainer, args: Namespace):
    train_args = {}
    if args.resume_from_checkpoint:
        train_args['resume_from_checkpoint'] = True
    start_time = time.time()
    trainer.train(**train_args)
    print('Training time:', time.time() - start_time)


def test(
    # trainer: Seq2SeqTrainer, 
    model,
    tokenizer,
    dataset: AfqmcSeq2SeqDataset,
    output_dir: Path, args: Namespace):
    '''Perform prediction (test) on given dataset'''
    preds, result = utils_seq2seq.predict(
        # trainer, 
        model, tokenizer,
        dataset, args)
    print(result)

    # Save
    output_dir.mkdir(exist_ok=True, parents=True)
    utils.dump_json(result, output_dir / 'result.json')
    preds_text = [tokenizer.decode(ids) for ids in preds]  # (N, ), list of str
    utils.dump_str(preds, output_dir / 'preds.txt')
    utils.dump_json(preds_text, output_dir / 'preds_text.json', indent=2)


def test_all(
    # trainer: Seq2SeqTrainer, 
    model,
    tokenizer, data_dir: Path):
    for test_phase in ['test_clean', 'test_noisy_1', 'test_noisy_2', 'test_noisy_3']:
        start_time = time.time()
        print('\nTesting phase:', test_phase)
        data = utils_seq2seq.get_dataset(data_dir, test_phase, tokenizer=tokenizer)
        # test(trainer, data, output_dir / test_phase, args)
        test(model, tokenizer, data, output_dir / test_phase, args)
        print('test_run_time:', time.time() - start_time)


args = parse_args()
output_dir = Path(args.output_dir)
data_dir = Path(args.data_dir)

utils.set_seed(args.seed)
utils.dump_args(args, output_dir / 'train_args.json')

# Get model
model = MT5ForConditionalGeneration.from_pretrained(args.model_path)
tokenizer = MT5Tokenizer.from_pretrained(args.model_path)
print('# parameters:', utils.get_param_count(model))

# Train and test
trainer = utils_seq2seq.get_trainer(model, tokenizer, data_dir, output_dir, args)
train(trainer, args)
test_all(model, tokenizer, data_dir)