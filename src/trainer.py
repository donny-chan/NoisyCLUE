from pathlib import Path
from argparse import Namespace
from typing import Any
from time import time

import torch
from torch.nn import Module
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler

from utils import dump_args, get_param_count, dump_json, load_json


def get_adamw(model: Module,lr: float, weight_decay: float) -> AdamW:
    """Prepare optimizer and schedule (linear warmup and decay)"""
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        betas=(0.9, 0.98),  # according to RoBERTa paper
        lr=lr)
    return optimizer


def get_linear_scheduler(optimizer, warmup_ratio: float, num_opt_steps: int):
    scheduler = get_scheduler(
        'linear',
        optimizer, 
        num_warmup_steps=int(warmup_ratio * num_opt_steps),
        num_training_steps=num_opt_steps)
    return scheduler


class TrainArgs:
    attrs = [
        'batch_size', 
        'grad_acc_steps', 
        'num_epochs', 
        'lr', 
        'weight_decay', 
        'warmup_ratio',
    ]

    def __init__(
        self, 
        batch_size: int=2,
        grad_acc_steps: int=1,
        num_epochs: int=1,
        log_interval: int=1,
        lr: float=1e-4,
        weight_decay: float=0.01,
        warmup_ratio: float=0.1):
        self.batch_size = batch_size
        self.grad_acc_steps = grad_acc_steps
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
    
    def from_args(self, args):
        for name in self.attrs:
            setattr(self, name, getattr(args, name))

    def to_dict(self) -> dict:
        return {key: getattr(self, key) for key in self.attrs}

    def from_dict(self):
        for name in self.attrs:
            setattr(self, name, getattr(self, name))

    def save(self, file: str):
        dump_json(self.to_dict(), file)

    def load(self, file: str):
        self.from_dict(load_json(file))


class Trainer:
    def log(self, *args, **kwargs):
        print(*args, **kwargs, flush=True)
        print(*args, **kwargs, file=self.train_log_writer, flush=True)

    def __init__(
        self,
        model: Module, 
        output_dir: str,
        batch_size: int=2,
        num_epochs: int=2,
        grad_acc_steps: int=1,
        log_interval: int=1,
        lr: float=1e-4):

        self.model = model
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.grad_acc_steps = grad_acc_steps
        self.log_interval = log_interval
        self.lr = lr
        self.setup_output_dir()

    def setup_output_dir(self):
        print('Setting up output dir...', flush=True)
        self.train_log_writer = open(
            self.output_dir / 'train.log', 'w', encoding='utf8')

    def setup_optimizer_and_scheuler(
        self, 
        lr: float,
        num_opt_steps: int, 
        weight_decay: float=0.01,
        warmup_ratio: float=0.1):
        '''
        Setup optimizer and scheduler.
        Will set `self.optimizer` and `self.scheduler`.
        '''
        self.optimizer = get_adamw(self.model, lr, weight_decay)
        self.scheduler = get_linear_scheduler(
            self.optimizer, warmup_ratio, num_opt_steps)

    def train(
        self, 
        train_dataset: Dataset, 
        dev_dataset: Dataset,
        ):
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizer
        num_opt_steps = len(train_dataloader) // self.grad_acc_steps * self.num_epochs
        self.setup_optimizer_and_scheuler(self.lr, num_opt_steps)

        self.global_cur_step = 0
        self.train_start_time = time()

        self.log(f'# params {get_param_count(self.model)}')
        self.log('*** Start training ***')
        self.log(f'batch size: {self.batch_size}')
        self.log(f'# grad acc steps: {self.grad_acc_steps}')
        self.log(f'# opt steps: {num_opt_steps}')
        self.log(f'# epochs: {self.num_epochs}')
        self.log(f'# train features: {len(train_dataset)}')
        self.log(f'# train steps: {len(train_dataloader)}')
        self.log(f'# dev features: {len(dev_dataset)}')

        for ep in range(self.num_epochs):
            # Learn
            self.train_epoch(train_dataloader, ep)

            # Validate
            checkpoint_dir = self.output_dir / f'checkpoint-{ep}'
            eval_output = self.evaluate(dev_dataset, 'dev', checkpoint_dir)
            self.save_checkpoint(checkpoint_dir)
            result = eval_output['result']
            preds = eval_output['preds']
            dump_json(result, checkpoint_dir / f'eval_result.json')
            dump_json(preds, checkpoint_dir / f'preds.json')

        self.log('*** End training ***')
        return self.get_train_result()

    def get_train_result(self) -> dict:
        return {
            'time_elapsed': time() - self.train_start_time,
        }

    def train_epoch(self, dataloader: DataLoader, epoch: int):
        self.train_dataloader = dataloader
        self.model.train()
        print(f'*** Start training epoch {epoch} ***', flush=True)
        for step, batch in enumerate(dataloader):
            self.train_step(step, batch)
        print(f'*** End training epoch {epoch} ***', flush=True)

    def eval_loop(self,
        dataset: Dataset,
        desc: str):
        '''Evaluate model on dataloader'''
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        self.eval_start_time = time()

        self.log(f'*** Start evaluating {desc} ***')
        self.log(f'# features: {len(dataset)}')
        self.log(f'# steps: {len(dataloader)}')

        for step, batch in enumerate(dataloader):
            self.eval_step(step, batch)
        
        self.log(f'*** Done evaluating {desc} ***')
        
        self.eval_end_time = time()
        self.eval_time_elapsed = self.eval_end_time - self.eval_start_time

    def save_checkpoint(self, output_dir: Path):
        torch.save(self.optimizer, output_dir / 'optimizer.pt')
        torch.save(self.model.state_dict(), output_dir / 'pytorch_model.bin')
        torch.save(torch.cuda.get_rng_state(), output_dir / 'rng_state.pth')
    
    def load_model(self, file: Path):
        '''Load model from file'''
        state_dict = torch.load(file, map_location='cpu')
        self.model.load_state_dict(state_dict)

    def load_best_checkpoint(self):
        '''Load best checkpoint in `self.output_dir` by dev loss'''
        min_loss = float('inf')
        best_checkpoint_dir = None
        for checkpoint_dir in sorted(self.output_dir.glob('checkpoint-*')):
            if not checkpoint_dir.is_dir():
                continue
            result_file = checkpoint_dir / 'eval_result.json'
            if not result_file.exists():
                continue
            result = load_json(result_file)
            loss = result['loss']
            if loss < min_loss:
                min_loss = loss
                best_checkpoint_dir = checkpoint_dir
        best_model_file = best_checkpoint_dir / 'pytorch_model.bin'
        self.log(f'Best model: {best_model_file}')
        self.load_model(best_model_file)

    '''
    Below are virtual functions required to implement
    '''

    def train_step(self, step: int, batch: dict):
        raise NotImplementedError

    def eval_step(self, step: int, batch: dict):
        raise NotImplementedError

    def evaluate(self, model: Module, dataset: Dataset, output_dir: Path, desc: str) -> dict:
        raise NotImplementedError

