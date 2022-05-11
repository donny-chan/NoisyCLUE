from pathlib import Path

from trainer import Trainer
from .data import NmtDataset


class NmtTrainer(Trainer):
    def eval_step(self, step: int, batch: dict):
        # Forward
        outputs = self.model(**batch)

        # Gather outputs
        loss = outputs.loss
        self.total_loss += loss.item()


    def evaluate(self,
        dataset: NmtDataset,
        desc: str='dev',
        output_dir: Path=None) -> dict:
        pass