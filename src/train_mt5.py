import json
import os
from pathlib import Path
from argparse import Namespace

from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from transformers.trainer import (
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    EvalPrediction)
import numpy as np
import torch

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


def get_test_acc(preds: np.array, labels: np.array):
    '''
    preds: (#examples, seq_len, vocab_size)
    labels: (#examples)
    '''
    preds = np.argmax(preds, axis=2) # (N, seq_len)
    preds = [verbalizer[p] for p in preds]

    print(preds.shape, labels.shape)
    assert preds.shape == labels.shape
    count = len(labels)
    correct = 0
    for i in range(count):
        if preds[i] == labels[i]:
            correct += 1
    return correct / count


def _get_dataset(file: Path, phase: str, **kwargs) -> AfqmcSeq2SeqDataset:
    return AfqmcSeq2SeqDataset(file, phase, **kwargs)


def get_dataset(data_dir, phase, **kwargs) -> AfqmcSeq2SeqDataset:
    kwargs['num_examples'] = 256  # NOTE: for debugging
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
    
    train_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True, # TODO: remove on release
        do_train=True,
        do_predict=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc_steps,
        # Move predictions to CPU often because vocab is very large.
        eval_accumulation_steps=16,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=lr,
        num_train_epochs=num_epochs,
        lr_scheduler_type='linear',
        warmup_ratio=warmup_ratio,
        logging_first_step=True,
        logging_steps=10,
        disable_tqdm=True,
        seed=0,
    )
    trainer = Trainer(
        model,
        train_args,
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset,
    )
    trainer.preprocess_logits_for_metrics = preprocess_logits_for_metrics,
    return trainer


def train(trainer: Trainer, args: Namespace):
    train_args = {}
    train_args['resume_from_checkpoint'] = True
    # if args.resume_from_checkpoint:
    #     train_args['resume_from_checkpoint'] = True
    trainer.train(**train_args)


def predict(trainer: Trainer, dataset: AfqmcSeq2SeqDataset, output_dir: Path):
    result = trainer.predict(dataset)
    preds = np.argmax(result.predictions[1], axis=2)  # (N, seq_len, V) -> (N, seq_len)
    pred_texts = [tokenizer.decode(ids) for ids in preds]  # (N, ), list of str

    # Save
    output_dir.mkdir(exist_ok=True, parents=True)
    utils.dump_str(list(preds), output_dir / 'preds.txt')
    utils.dump_json(pred_texts, output_dir / 'preds_text.json')

    # Calculate scores
    labels = result.label_ids # (N, seq_len=4)
    num_examples = len(preds)
    num_correct = 0
    print(labels.shape)
    print(preds.shape)
    for i in range(num_examples):
        label = labels[i]
        pred = preds[i]
        if label == pred:
            num_correct += 1
    acc = num_correct / num_examples
    loss = result.metrics['test_loss']
    result_dict = {
        'acc': acc,
        'loss': loss,
    }

    utils.dump_json(result_dict, output_dir / 'result.json')
    print('result:', result_dict)


# model_path = 'google/mt5-base'
MODEL_PATH = 'google/mt5-small'
args = parse_args()
output_dir = Path(args.output_dir)
data_dir = Path(args.data_dir)
os.environ["WANDB_DISABLED"] = "true"

utils.set_seed(0)
utils.dump_args(args, output_dir / 'train_args.json')
print(json.dumps(vars(args), indent=2, ensure_ascii=False))

# Get model
model = MT5ForConditionalGeneration.from_pretrained(MODEL_PATH)
tokenizer = MT5Tokenizer.from_pretrained(MODEL_PATH)
model = model.cuda()

# Train
trainer = get_trainer(model, tokenizer, data_dir, output_dir, args)
train(trainer, args)

# Test
def test_all(trainer, tokenizer, data_dir):
    for test_phase in ['test_clean', 'test_noisy_1', 'test_noisy_2', 'test_noisy_3']:
        print('\nTesting phase:', test_phase)
        data = get_dataset(data_dir, test_phase, tokenizer=tokenizer)
        predict(trainer, data, output_dir / test_phase)

test_all(trainer, tokenizer, data_dir)