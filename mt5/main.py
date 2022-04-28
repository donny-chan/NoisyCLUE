import json
import os
from pathlib import Path

from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from transformers.trainer import Trainer, TrainingArguments, EvalPrediction
import torch
import numpy as np

from data.afqmc import AfqmcDataset

# model_path = 'google/mt5-base'
output_dir = Path('results/afqmc')
model_path = 'google/mt5-small'
data_dir = Path('../data/afqmc/split')
os.environ["WANDB_DISABLED"] = "true"

model = MT5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = MT5Tokenizer.from_pretrained(model_path)
model = model.cuda()

def get_acc(preds, labels) -> float:
    return np.mean(np.argmax(preds, axis=1) == labels)

def get_test_acc(output) -> float:
    return np.mean(np.argmax(output.predictions[0], axis=2) == output.label_ids)

def compute_metrics(eval_pred: EvalPrediction) -> dict:
    acc = get_acc(eval_pred.predictions, eval_pred.label_ids)
    return {'acc': acc}

def get_dataset(data_dir, phase, tokenizer) -> AfqmcDataset:
    return AfqmcDataset(data_dir / f'{phase}.json', phase, tokenizer, 256)


def get_trainer(model: MT5ForConditionalGeneration, tokenizer: MT5Tokenizer,
                data_dir: Path, output_dir: Path) -> Trainer:
    train_dataset = get_dataset(data_dir, 'train', tokenizer=tokenizer)
    eval_dataset = get_dataset(data_dir, 'dev', tokenizer=tokenizer)
    
    # Hyperparameters
    batch_size = 4
    grad_acc_steps = 16
    num_epochs = 4
    warmup_ratio = 0.1
    lr = 2e-4
    
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
    pred_out = trainer.predict(dataset)
    preds = pred_out.predictions[1]
    print('\nTest result:')
    print('loss:', pred_out.metrics['test_loss'])
    print('acc:', get_acc(preds, pred_out.label_ids))
    json.dump(preds, open(output_dir / 'predictions.json', 'w'))
    pred_ids = np.argmax(preds, axis=2)
    pred_texts = [tokenizer.decode(ids) for ids in pred_ids]
    json.dump(pred_texts, open(output_dir / 'pred_texts.json', 'w'))
    
    

trainer = get_trainer(model, tokenizer, data_dir)

clean_test_data = get_dataset(data_dir, 'test', tokenizer=tokenizer)
noisy_test_data = get_dataset(data_dir, 'noisy_test', tokenizer=tokenizer)

predict(trainer, clean_test_data, output_dir / 'test_clean')
predict(trainer, noisy_test_data, output_dir / 'test_noisy')