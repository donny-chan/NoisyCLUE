from argparse import Namespace
from pathlib import Path
from time import time

import torch
from torch.optim import AdamW
from torch.nn import Module
from torch.utils.data import DataLoader

from utils import dump_args, dump_jsonl
from trainer import Trainer
from .data import CluenerDataset


class CluenerTrainer(Trainer):
    def train_log(self, cur_loss):
        state = {
            'ep': self.global_cur_step / len(self.train_dataloader),
            'lr': self.scheduler.get_last_lr()[0],
            'loss': cur_loss.item(),
            'time_elapsed': time() - self.train_start_time,
        }
        self.log(state)

    def train_step(self, step: int, batch: dict):
        # Forward
        outputs = self.model(**batch)
        loss = outputs.loss
        self.global_cur_step += 1

        # Backward
        if self.global_cur_step % self.grad_acc_steps == 0:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        # Log
        if self.global_cur_step % self.log_interval == 0:
            self.train_log(loss)

    def eval_step(self, step: int, batch: dict):
        # Forward
        outputs = self.model(**batch)

        # Gather outputs
        loss = outputs.loss
        self.total_loss += loss.item()
        logits = outputs.logits       # (B, N, C)
        pred_ids = logits.argmax(-1)  # (B, N)
        self.all_preds += pred_ids.tolist()
        self.all_labels += batch['labels'].tolist()

        self.cur_eval_step = step

    def evaluate(self, 
        dataset: CluenerDataset, 
        desc: str='dev', 
        output_dir: Path=None) -> dict:
        '''
        Evaluate the model on the dataset.
        '''
        # Result to gather
        self.total_loss = 0
        self.all_labels = []  # (# examples, N)
        self.all_preds = []   # (# examples, N)

        # Evaluation
        self.eval_loop(dataset, desc)
        
        # Process gathered result
        output_dir.mkdir(exist_ok=True, parents=True)
        id2label = dataset.get_id2label()
        for i, pred in enumerate(self.all_preds):
            self.all_preds[i] = [id2label[x] for x in pred]

        result = {
            'acc': self.get_metrics(self.all_preds, self.all_labels)['acc'],
            'loss': self.total_loss / self.cur_eval_step,
            'time_elapsed': time() - self.eval_start_time,
        }
        self.log(result)
        return {
            'result': result,
            'preds': self.all_preds,
        }

    def get_metrics(self, preds, labels):
        assert len(preds) == len(labels)
        count = len(preds)
        correct = 0
        for i in range(count):
            if preds[i] == labels[i]:
                correct += 1
        return {
            'acc': correct / count,
        }