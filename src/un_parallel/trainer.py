from pathlib import Path
from time import time

from torch.utils.data import DataLoader
from transformers.optimization import Adafactor, AdafactorSchedule, get_constant_schedule_with_warmup, get_scheduler

from trainer import Trainer
from utils import dump_json
from .data import UNParallelZhEnIterableDataset


class UNParallelTrainer(Trainer):
    def get_train_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size)

    def setup_optimizer_and_scheuler(
        self, 
        lr: float, 
        num_opt_steps: int, 
        weight_decay: float = 0.01, 
        warmup_ratio: float = 0.1):
        self.log('Setting up Adafactor optimizer and scheduler...')
        self.optimizer = Adafactor(
            self.model.parameters(), 
            scale_parameter=False, 
            relative_step=True, 
            warmup_init=True, 
            lr=None)
        # lrs = get_constant_schedule_with_warmup(self.optimizer, 100)
        self.optimizer = Adafactor(
            self.model.parameters(), 
            scale_parameter=False, 
            relative_step=False, 
            warmup_init=False, 
            lr=lr)
        # self.scheduler = AdafactorSchedule(self.optimizer)

        # Will use the LR in optimizer
        self.scheduler = get_constant_schedule_with_warmup(self.optimizer, 100)

    def eval_step(self, step: int, batch: dict):
        # Forward
        outputs = self.model(**batch)


        # Gather outputs
        loss = outputs.loss
        self.total_loss += loss.item()
        logits = outputs.logits
        preds = logits.argmax(dim=-1)
        self.all_preds += preds.tolist()
        self.all_labels += batch['labels'].tolist()

        if (step + 1) % self.log_interval == 0:
            self.log({
                'step': step + 1,
                'loss': self.total_loss / (step + 1),
                'time_elapsed': time() - self.eval_start_time,
            })


    def evaluate(self,
        dataset: UNParallelZhEnIterableDataset,
        output_dir: Path,
        desc: str='dev',
        ) -> dict:
        
        # Define variable for gathering results of evaluation
        self.total_loss = 0
        self.all_preds = []
        self.all_labels = []

        dataset.reset()
        self.eval_loop(dataset, desc)  # Must call this, this will call `eval_step`

        dump_json(self.all_preds, 'preds.json')
        dump_json(self.all_labels, 'labels.json')

        # Process gathered result
        output_dir.mkdir(exist_ok=True, parents=True)
        result = {
            'loss': self.total_loss / self.num_eval_steps,
            'time_elapsed': time() - self.eval_start_time,
        }
        self.log(result)
        return {
            'result': result,
            'preds': self.all_preds
        }
        