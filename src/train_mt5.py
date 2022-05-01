import torch
assert torch.cuda.is_available()

import time
import json
import os
from pathlib import Path
from argparse import Namespace

from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.trainer import (
    Trainer, 
    TrainingArguments, 
    EvalPrediction,
)
import numpy as np
from torch.utils.data import DataLoader

from data.afqmc import AfqmcSeq2SeqDataset
import utils
from arguments import parse_args


def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    '''
    logits: (N, seq_len, vocab_size)
    labels: (N, seq_len)
    '''
    return torch.argmax(logits, dim=2)  # (N, seq_len, vocab_size) -> (N, seq_len)


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    acc = utils.get_acc(eval_pred.predictions, eval_pred.label_ids)
    return {'acc': acc}


def get_test_acc(preds: torch.Tensor, labels: torch.Tensor) -> float:
    '''
    preds: (#examples, seq_len)
    labels: (#examples, seq_len)
    '''
    assert preds.size() == labels.size()
    eq = torch.eq(preds, labels)
    count = len(labels)
    correct = 0
    for i in range(count):
        if eq[i].all():
            correct += 1
    return correct / count


def _get_dataset(file: Path, phase: str, **kwargs) -> AfqmcSeq2SeqDataset:
    return AfqmcSeq2SeqDataset(file, phase, **kwargs)


def get_dataset(data_dir: Path, phase: str, **kwargs) -> AfqmcSeq2SeqDataset:
    # kwargs['num_examples'] = 256  # NOTE: for debugging
    return _get_dataset(data_dir / f'{phase}.json', phase, **kwargs) 


def get_trainer(
    model: MT5ForConditionalGeneration, tokenizer: MT5Tokenizer, data_dir: Path,
    output_dir: Path, args: Namespace) -> Trainer:

    train_dataset = get_dataset(data_dir, 'train', tokenizer=tokenizer)
    eval_dataset = get_dataset(data_dir, 'dev', tokenizer=tokenizer)
    
    # Hyperparameters
    batch_size = args.batch_size
    grad_acc_steps = args.grad_acc_steps
    num_epochs = args.num_epochs
    warmup_ratio = 0.1
    lr = args.lr
    
    train_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True, # TODO: remove on release
        do_train=True,
        do_predict=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc_steps,
        # Move predictions to CPU often because vocab is very large.
        eval_accumulation_steps=4,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=lr,
        num_train_epochs=num_epochs,
        lr_scheduler_type='linear',
        warmup_ratio=warmup_ratio,
        report_to='none',
        logging_first_step=True,
        logging_steps=args.log_interval,
        disable_tqdm=True,
        seed=0,
    )
    trainer = Seq2SeqTrainer(
        model,
        train_args,
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset,
    )
    # trainer.preprocess_logits_for_metrics = preprocess_logits_for_metrics,
    return trainer


def train(trainer: Trainer, args: Namespace):
    train_args = {}
    # train_args['resume_from_checkpoint'] = True
    if args.resume_from_checkpoint:
        train_args['resume_from_checkpoint'] = True
    start_time = time.time()
    trainer.train(**train_args)
    time_elapsed = time.time() - start_time
    print('Training time:', time_elapsed)


def predict(trainer: Trainer, dataset: AfqmcSeq2SeqDataset, output_dir: Path, 
        args: Namespace):
    trainer.model.eval()

    def collate_fn(examples: list):
        '''Each element in `examples` is a dict from str to list.'''
        batch = {}
        for key in examples[0].keys():
            batch[key] = torch.tensor([x[key] for x in examples])
        return batch

    def prediction_step(batch: dict) -> tuple:
        return trainer.prediction_step(trainer.model, inputs=batch, 
            prediction_loss_only=False)
    
    def compute_metrics(preds: torch.Tensor, labels: torch.Tensor) -> dict:
        return {
            'acc': get_test_acc(preds, labels),
        }


    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    total_loss = 0
    acc = 0
    num_steps = 0

    model = trainer.model
    all_preds = []
    
    for step, batch in enumerate(dataloader):
        loss, logits, labels = prediction_step(batch)
        logits = logits[0]
        preds = torch.argmax(logits, dim=2)  # (N, seq_len, vocab_size) -> (N, seq_len)
        all_preds += list(preds.cpu().numpy())

        total_loss += loss.item()
        # result = compute_metrics(preds, labels)
        # acc += result['acc']
        acc += get_test_acc(preds, labels)
        num_steps += 1
    # Average
    acc /= num_steps
    loss = total_loss / num_steps
    result = {
        'acc': acc,
        'loss': loss,
    }
    print('result:', result)
    # Save
    output_dir.mkdir(exist_ok=True, parents=True)
    utils.dump_json(result, output_dir / 'result.json')
    pred_texts = [tokenizer.decode(ids) for ids in all_preds]  # (N, ), list of str
    utils.dump_str(all_preds, output_dir / 'preds.txt')
    utils.dump_json(pred_texts, output_dir / 'preds_text.json')


MODEL_PATH = 'google/mt5-base'
# MODEL_PATH = 'google/mt5-small'
args = parse_args()
output_dir = Path(args.output_dir)
data_dir = Path(args.data_dir)
# os.environ["WANDB_DISABLED"] = "true"

utils.set_seed(0)
utils.dump_args(args, output_dir / 'train_args.json')
# print(json.dumps(vars(args), indent=2, ensure_ascii=False))

# Get model
model = MT5ForConditionalGeneration.from_pretrained(MODEL_PATH)
tokenizer = MT5Tokenizer.from_pretrained(MODEL_PATH)
# model = torch.nn.DataParallel(model, device_ids=[0, 1])
# model = model.cuda()
print('# parameters:', utils.get_param_count(model))

# Train
trainer = get_trainer(model, tokenizer, data_dir, output_dir, args)
train(trainer, args)

# Test
def test_all(trainer: Trainer, tokenizer, data_dir: Path):
    for test_phase in ['test_clean', 'test_noisy_1', 'test_noisy_2', 'test_noisy_3']:
        print('\nTesting phase:', test_phase)
        data = get_dataset(data_dir, test_phase, tokenizer=tokenizer)
        predict(trainer, data, output_dir / test_phase, args)

test_all(trainer, tokenizer, data_dir)