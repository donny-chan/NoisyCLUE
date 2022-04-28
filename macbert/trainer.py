from pathlib import Path
from datetime import datetime

from transformers.trainer import Trainer, TrainingArguments
from torch import nn
from torch.utils.data import Dataset, DataLoader

class Trainer:
    def __init__(self, model: nn.Module, tokenizer, data_dir: Path):
        self.tokenizer = tokenizer
        self.train_file = data_dir / 'train.json'
        self.dev_file = data_dir / 'dev.json'
        
        self.model = model
        self.train_args = TrainingArguments(
            'results/temp', 
            overwrite_output_dir=True, # TODO: remove on release
            do_train=True,
            do_predict=True,
            evaluation_strategy='epoch',
            learning_rate=2e-5,
            num_train_epochs=2,
            lr_scheduler_type='linear',
            warmup_ratio=0.1,
            logging_dir='results/' + datetime.now().strftime("%y%m%d%H%M%S"),
            logging_first_step=True,
            logging_steps=1,
            seed=0,
        )

        self.trainer = Trainer(
            model,
            self.train_args,
            train_dataset=train_dataset, 
            eval_dataset=eval_dataset,
        )

    def train(model):

        trainer.train(num_train_epochs=4)


    # Test on clean data
    def test(model: nn.Module, test_dataset: Dataset):
        # Test on noisy data
        trainer.predict(clean_data)