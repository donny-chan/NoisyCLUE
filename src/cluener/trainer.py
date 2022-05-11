from pathlib import Path
from time import time

from trainer import Trainer
from .data import CluenerDataset
from .evaluate import get_metrics
from utils import dump_json


class CluenerTrainer(Trainer):
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
        # TODO: remove on release
        dump_json(self.all_preds, 'preds.json')  
        dump_json(self.all_labels, 'labels.json')

        id2label = dataset.get_id2label()
        metrics = get_metrics(self.all_labels, self.all_preds, id2label)

        result = {
            'prec': metrics['prec'],
            'f1': metrics['f1'],
            'recall': metrics['recall'],
            'loss': self.total_loss / self.num_eval_steps,   # This must be provided for choosing best model
            'time_elapsed': time() - self.eval_start_time,
        }
        self.log(result)
        return {
            'result': result,
            'preds': self.all_preds,
        }
