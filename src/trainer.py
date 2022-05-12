from pathlib import Path
from argparse import Namespace
from time import time
import json
from typing import Union

import torch
from torch.nn import Module
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler

from utils import get_param_count, dump_json, load_json


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
        lr=lr,
        eps=1e-8)
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
    '''
    Simple trainer that supports:
    - Easy interface for training and evaluating any model
    - saving and resuming.
    '''
    LAST_EPOCH_FILE = 'last_epoch.txt'
    LAST_CKPT_FILE = 'last_ckpt.txt'
    TRAIN_LOG_FILE = 'train.log'

    def log(self, *args, **kwargs):
        print(*args, **kwargs, flush=True)
        if self.do_log_to_file and self.train_log_writer:
            print(*args, **kwargs, file=self.train_log_writer, flush=True)

    def __init__(
        self,
        model: Module, 
        output_dir: str,
        batch_size: int=2,
        num_epochs: int=2,
        grad_acc_steps: int=1,
        lr: float=1e-4,
        log_interval: int=1,
        eval_interval: int=500,
        eval_strategy: str='step',
        do_log_to_file: bool=True,
        log_file: str=None,
        ):

        self.model = model
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.grad_acc_steps = grad_acc_steps
        self.lr = lr
        self.log_interval = log_interval
        self.eval_strategy = eval_strategy
        self.eval_interval = eval_interval
        self.do_log_to_file = do_log_to_file
        self.log_file = log_file
        # self.save_strategy = eval_strategy
        # self.save_interval = eval_interval
        self.setup_output_dir()

    def train_log(self, cur_loss):
        '''Called every `log_interval` steps during training.'''
        state = {
            'step': self.cur_train_step,
            'ep': self.cur_train_step / len(self.train_dataloader),
            'lr': self.scheduler.get_last_lr()[0],
            'loss': cur_loss.item(),
            'time_elapsed': time() - self.train_start_time,
        }
        self.log(state)

    def setup_output_dir(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)
        if self.do_log_to_file:
            if self.log_file:
                log_file = self.output_dir / self.log_file
            else:
                log_file = self.output_dir / self.TRAIN_LOG_FILE
            self.train_log_writer = open(log_file, 'w', encoding='utf8')
        self.log('Setting up trainer with the following arguments:')
        self.log('--------- Training args ---------')
        for key in [
            'output_dir', 
            'batch_size', 
            'num_epochs', 
            'grad_acc_steps', 
            'lr',
            'log_interval', 
            'eval_interval',
            'eval_strategy',
        ]:
            self.log(f'{key:>16}: {getattr(self, key)}')
        self.log('---------------------------------')

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
        self.log(f'Setting up AdamW optimizer and linear scheduler...')
        self.optimizer = get_adamw(self.model, lr, weight_decay)
        self.scheduler = get_linear_scheduler(
            self.optimizer, warmup_ratio, num_opt_steps)

    ''' Saving and Loading '''

    def get_last_ckpt_file(self) -> Path:
        return self.output_dir / self.LAST_CKPT_FILE

    def load_last_ckpt(self) -> dict:
        return json.load(self.get_last_ckpt_file().open('r'))

    def set_epoch_and_step(self, save_settings: dict) -> int:
        self.cur_epoch = save_settings['epoch']
        self.cur_train_step = save_settings['step']
        self.eval_strategy = save_settings['save_strategy']

    def get_epoch_and_step(self):
        return {
            'epoch': self.cur_epoch,
            'step': self.cur_train_step,
            'save_strategy': self.eval_strategy,
        }

    def dump_last_ckpt(self):
        with self.get_last_ckpt_file().open('w') as f:
            json.dump(self.get_epoch_and_step(), f)

    def load_last_ckpt_dir(self) -> Path:
        last_ckpt = self.load_last_ckpt()
        ckpt_id = last_ckpt[last_ckpt['save_strategy']]
        return self.output_dir / f'checkpoint-{ckpt_id}'

    def can_resume(self) -> bool:
        return self.get_last_ckpt_file().exists()

    def load_best_ckpt(self):
        '''Load best checkpoint in `self.output_dir` by dev loss'''
        min_loss = float('inf')
        best_ckpt_dir = None
        for ckpt_dir in sorted(self.output_dir.glob('checkpoint-*')):
            if not ckpt_dir.is_dir():
                continue
            result_file = ckpt_dir / 'eval_result.json'
            if not result_file.exists():
                continue
            result = load_json(result_file)
            loss = result['loss']
            if loss < min_loss:
                min_loss = loss
                best_ckpt_dir = ckpt_dir
        self.load_ckpt(best_ckpt_dir)

    def resume(self):
        '''Resume training from last checkpoint'''
        self.log(f'*** Resuming training from last checkpoint ***')
        last_ckpt_dir = self.load_last_ckpt_dir()
        # self.cur_epoch = self.load_last_epoch())
        # self.load_ckpt(self.output_dir / f'checkpoint-{self.cur_epoch}')
        self.load_ckpt(last_ckpt_dir)
        self.log(f'*** Resumed training from end of epoch {self.cur_epoch} ***')

    def get_cur_ckpt_dir(self) -> Path:
        if self.eval_strategy == 'step':
            return self.output_dir / f'checkpoint-{self.cur_train_step}'
        else:
            return self.output_dir / f'checkpoint-{self.cur_epoch}'

    def save_ckpt(self):
        '''
        Save current checkpoint to `{output_dir}/checkpoint-{ckpt_num}`
        '''
        ckpt_dir = self.get_cur_ckpt_dir()
        self.log(f'*** Saving checkpoint to {ckpt_dir} ***')
        ckpt_dir.mkdir(exist_ok=True, parents=True)
        torch.save(self.model.state_dict(), ckpt_dir / 'pytorch_model.bin')
        ckpt = {
            'cur_epoch': self.cur_epoch,
            'global_cur_step': self.cur_train_step,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'train_elapsed': time() - self.train_start_time,
        }
        torch.save(ckpt, ckpt_dir / 'ckpt.bin')

        self.dump_last_ckpt()
        self.log(f'*** Done saving checkpoint to {ckpt_dir} ***')
    
    def load_model(self, file: Path):
        '''Load model from file'''
        state_dict = torch.load(file, map_location='cpu')
        self.model.load_state_dict(state_dict)

    def load_ckpt(self, ckpt_dir: Path):
        '''Load checkpoint'''
        self.log(f'*** Loading checkpoint from {ckpt_dir} ***')
        self.load_model(ckpt_dir / 'pytorch_model.bin')
        ckpt = torch.load(ckpt_dir / 'ckpt.bin')

        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.train_start_time = time() - ckpt['train_elapsed']
        self.cur_epoch = ckpt['cur_epoch']
        self.cur_train_step = ckpt['global_cur_step']
        self.log(f'*** Done loading checkpoint from {ckpt_dir} ***')

    ''' End Saving and Loading '''

    def validate(self, dev_dataset):
        '''
        Call this on validation, NOT on test!
        
        Will output results to current checkpoint dir.
        '''
        ckpt_dir = self.get_cur_ckpt_dir()

        # Actual evaluation
        eval_output = self.evaluate(dev_dataset, ckpt_dir, 'dev')
        
        result = eval_output['result']
        preds = eval_output['preds']
        dump_json(result, ckpt_dir / f'eval_result.json')
        dump_json(preds, ckpt_dir / f'preds.json')

    def get_train_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train(
        self, 
        train_dataset: Union[Dataset, DataLoader], 
        dev_dataset: Dataset,
        resume: bool=True,
        ):
        self.dev_dataset = dev_dataset
        self.train_dataset = train_dataset
        train_dataloader = self.get_train_dataloader(train_dataset)

        # Optimizer
        num_opt_steps = len(train_dataloader) // self.grad_acc_steps * self.num_epochs
        self.setup_optimizer_and_scheuler(self.lr, num_opt_steps)

        # Handle resumption
        if resume and self.can_resume():
            self.resume()
            # TODO: This is incorrect?
            self.validate(dev_dataset)
            self.cur_epoch += 1
        else:
            self.cur_epoch = 0
            self.cur_train_step = 0
            self.train_start_time = time()

        self.log('\n*** Start training ***')
        self.log(f'batch size: {self.batch_size}')
        self.log(f'# grad acc steps: {self.grad_acc_steps}')
        self.log(f'# opt steps: {num_opt_steps}')
        self.log(f'# epochs: {self.num_epochs}')
        self.log(f'# train features: {len(train_dataset)}')
        self.log(f'# train steps: {len(train_dataloader)}')
        self.log(f'# dev features: {len(dev_dataset)}')
        self.log(f'# params {get_param_count(self.model)}')

        while self.cur_epoch < self.num_epochs:
            self.cur_epoch += 1
            self.train_epoch(train_dataloader)
            if self.eval_strategy == 'epoch' and self.cur_epoch % self.eval_interval == 0:
                self.save_ckpt()
                self.validate(dev_dataset)

        self.log('*** End training ***')
        return self.get_train_result()

    def get_train_result(self) -> dict:
        return {
            'time_elapsed': time() - self.train_start_time,
        }

    def train_epoch(self, dataloader: DataLoader):
        self.train_dataloader = dataloader
        self.model.train()
        self.log(f'*** Start training epoch {self.cur_epoch} ***')
        for step, batch in enumerate(dataloader):
            self.train_step(step, batch)
        self.log(f'*** End training epoch {self.cur_epoch} ***')

    def eval_loop(self,
        dataset: Dataset,
        desc: str):
        '''
        
        Evaluate model on dataloader
        '''
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        self.eval_start_time = time()
        self.num_eval_steps = len(dataloader)

        self.log(f'*** Start evaluating {desc} ***')
        self.log(f'# features: {len(dataset)}')
        self.log(f'# steps: {len(dataloader)}')

        for step, batch in enumerate(dataloader):
            self.eval_step(step, batch)
        
        self.log(f'*** Done evaluating {desc} ***')
        
        self.eval_end_time = time()
        self.eval_time_elapsed = self.eval_end_time - self.eval_start_time

    '''
    Below are virtual functions required to implement
    '''

    def backward(self, step: int, loss: Module):
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        

    def train_step(self, step: int, batch: dict):
        self.cur_train_step += 1

        # Forward
        outputs = self.model(**batch)
        loss = outputs.loss

        # Backward
        if self.cur_train_step % self.grad_acc_steps == 0:
            self.backward(step, loss)

        # Log
        if self.cur_train_step % self.log_interval == 0:
            self.train_log(loss)

        if self.eval_strategy == 'step' and self.cur_train_step % self.eval_interval == 0:
            self.save_ckpt()
            self.validate(self.dev_dataset)
            self.model.train()

    def eval_step(self, step: int, batch: dict):
        raise NotImplementedError

    def evaluate(self, model: Module, dataset: Dataset, output_dir: Path, desc: str) -> dict:
        '''
        Overriders of this must:

        - call `self.eval_loop`!!
        - return a dict with keys: 'result' and 'preds'.
        '''
        raise NotImplementedError

