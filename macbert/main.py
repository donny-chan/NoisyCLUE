import os
import json
from pathlib import Path
from datetime import datetime

from transformers import BertForSequenceClassification, BertTokenizer
from transformers.trainer import Trainer, TrainingArguments
import numpy as np

from data.afqmc import AfqmcDataset


def get_accuracy(preds: np.array, labels: np.array) -> float:
    return (np.argmax(preds, axis=1) == labels).mean()


def _get_dataset(data_dir: Path, phase: str, **kwargs) -> AfqmcDataset:
    return AfqmcDataset(data_dir / f'{phase}.json', phase, max_seq_len=512, **kwargs)


def get_trainer(model: BertForSequenceClassification, tokenizer: BertTokenizer,
                data_dir: Path, output_dir: Path):
    train_dataset = _get_dataset(data_dir, 'train', tokenizer=tokenizer)
    eval_dataset = _get_dataset(data_dir, 'dev', tokenizer=tokenizer)
    
    # Hyperparameters
    batch_size = 8
    grad_acc_steps = 16
    num_epochs = 8
    warmup_ratio = 0.1
    lr = 2e-5
    
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
        logging_steps=10,
        # disable_tqdm=True,
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
    result = trainer.predict(dataset)
    print('\nTest result:')
    print('loss:', result.metrics['test_loss'])
    print('acc:', get_accuracy(result.predictions, result.label_ids))
    json.dump(result.predictions, open(output_dir / 'predictions.json', 'w'), indent=2)


output_dir = Path('results/afqmc/macbert-B64-LR2e-05')
model_path = 'hfl/chinese-macbert-base'
data_dir = Path('../data/afqmc/split')
os.environ["WANDB_DISABLED"] = "true"

# Train
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path).cuda()
trainer = get_trainer(model, tokenizer, data_dir, output_dir)
trainer.train()

# Test
print('Preparing test data')
clean_data = _get_dataset(data_dir, 'test', tokenizer=tokenizer)
noisy_data = _get_dataset(data_dir, 'noisy_test', tokenizer=tokenizer)

print('Testing on clean data')
predict(trainer, clean_data, output_dir / 'test_clean')
print('Testing on noisy data')
predict(trainer, noisy_data, output_dir / 'test_noisy')