from pathlib import Path
from time import time

from torch.utils.data import DataLoader

from trainer import Trainer
from .data import NmtDataset


class NmtTrainer(Trainer):
    def get_train_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size)

    def eval_step(self, step: int, batch: dict):
        # Forward
        outputs = self.model(**batch)

        # Gather outputs
        loss = outputs.loss
        self.total_loss += loss.item()
        print(outputs)
        exit()


    def evaluate(self,
        dataset: NmtDataset,
        desc: str='dev',
        output_dir: Path=None
        ) -> dict:
        
        # Define variable for gathering results of evaluation
        self.total_loss = 0
        self.all_preds = []

        self.eval_loop(dataset, desc)  # Must call this, this will call `eval_step`

        # Process gathered result
        output_dir.mkdir(exist_ok=True, parents=True)
        result = {
            'preds': self.all_preds,
            'loss': self.total_loss / self.num_eval_steps,
            'time_elapsed': time() - self.eval_start_time,
        }
        self.log(result)
        return {
            'result': result,
            'preds': self.all_preds
        }
        