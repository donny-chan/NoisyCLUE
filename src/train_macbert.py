import os
import json
from pathlib import Path
from argparse import Namespace

from transformers import BertForSequenceClassification, BertTokenizer
from transformers.trainer import Trainer, TrainingArguments
import numpy as np
import torch

from data.afqmc import AfqmcDataset
import utils
from arguments import parse_args


def get_test_acc(preds: np.array, labels: np.array) -> float:
    return (np.argmax(preds, axis=1) == labels).mean()


def _get_dataset(file: Path, phase: str, **kwargs) -> AfqmcDataset:
    # kwargs['num_examples'] = 128  # for debugging
    return AfqmcDataset(file, phase, max_seq_len=512, **kwargs)


def  get_dataset(data_dir: Path, phase: str, **kwargs) -> AfqmcDataset:
    return _get_dataset(data_dir / f'{phase}.json', phase, **kwargs)


def get_trainer(model: BertForSequenceClassification, tokenizer: BertTokenizer,
                data_dir: Path, output_dir: Path, args: Namespace):
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
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=lr,
        num_train_epochs=num_epochs,
        lr_scheduler_type='linear',
        warmup_ratio=warmup_ratio,
        logging_first_step=True,
        logging_steps=args.log_interval,
        logging_strategy='steps',
        disable_tqdm=True,
        seed=0,
    )
    trainer = Trainer(
        model,
        train_args,
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset,
    )
    return trainer


def predict(trainer: Trainer, dataset: AfqmcDataset, output_dir: Path):
    def parse_result(result):
        return {'loss': result.metrics['test_loss'],
                'acc': get_test_acc(result.predictions, result.label_ids)}

    result = trainer.predict(dataset)
    result_dict = parse_result(result)
    print('Test result:')
    print('loss:', result_dict['loss'])
    print('acc:', result_dict['acc'])

    # Save result
    output_dir.mkdir(exist_ok=True, parents=True)
    preds = np.argmax(result.predictions, axis=1) # (N, C) -> (N)
    utils.dump_str(list(preds), output_dir / 'preds.txt')
    utils.dump_json(result_dict, output_dir / 'result.json')


def train(trainer: Trainer, args: Namespace):
    train_args = {}
    if args.resume_from_checkpoint:
        train_args['resume_from_checkpoint'] = True
    trainer.train(**train_args)

# Test
def test(trainer, tokenizer, data_dir):
    print('Preparing test data')
    for test_phase in ['test_clean', 'test_noisy_1', 'test_noisy_2', 'test_noisy_3']:
        print('\nTesting phase:', test_phase)
        data = get_dataset(data_dir, test_phase, tokenizer=tokenizer)
        predict(trainer, data, output_dir / test_phase)

# Setup
# MODEL_PATH = 'hfl/chinese-macbert-base'
args = parse_args()
output_dir = Path(args.output_dir)
data_dir = Path(args.data_dir)
os.environ["WANDB_DISABLED"] = "true"

utils.set_seed(0)
utils.dump_args(args, output_dir / 'train_args.json')

# Model
tokenizer = BertTokenizer.from_pretrained(args.model_path)
model = BertForSequenceClassification.from_pretrained(args.model_path).cuda()
print('# params:', utils.get_param_count(model))

# Train
trainer = get_trainer(model, tokenizer, data_dir, output_dir, args)
train(trainer, args)
test(trainer, tokenizer, data_dir)