import json
import os
from pathlib import Path
from argparse import Namespace

from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from transformers.trainer import Trainer, TrainingArguments, EvalPrediction
import numpy as np

from data.afqmc import AfqmcSeq2SeqDataset
import utils
from arguments import parse_args

def get_acc(preds, labels) -> float:
    return np.mean(np.argmax(preds, axis=1) == labels)

def get_test_acc(output) -> float:
    return np.mean(np.argmax(output.predictions[0], axis=2) == output.label_ids)

def compute_metrics(eval_pred: EvalPrediction) -> dict:
    acc = get_acc(eval_pred.predictions, eval_pred.label_ids)
    return {'acc': acc}

def get_dataset(data_dir, phase, tokenizer) -> AfqmcSeq2SeqDataset:
    return AfqmcSeq2SeqDataset(data_dir / f'{phase}.json', phase, tokenizer, 128)


def get_trainer(
    model: MT5ForConditionalGeneration, tokenizer: MT5Tokenizer, data_dir: Path,
    output_dir: Path, args: Namespace) -> Trainer:

    train_dataset = get_dataset(data_dir, 'train', tokenizer=tokenizer)
    eval_dataset = get_dataset(data_dir, 'dev', tokenizer=tokenizer)
    
    # Hyperparameters
    batch_size = args.batch_size
    grad_acc_steps = args.grad_acc_steps
    num_epochs = args.num_epochs
    warmup_ratio = 0.1
    lr = args.lr
    
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
        disable_tqdm=True,
        seed=0,
    )
    trainer = Trainer(
        model,
        train_args,
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset,
    )
    return trainer


def predict(trainer: Trainer, dataset: AfqmcSeq2SeqDataset, output_dir: Path):
    pred_out = trainer.predict(dataset)
    preds = pred_out.predictions[1]
    print('\nTest result:')
    print('loss:', pred_out.metrics['test_loss'])
    print('acc:', get_acc(preds, pred_out.label_ids))
    json.dump(preds, open(output_dir / 'predictions.json', 'w'))
    pred_ids = np.argmax(preds, axis=2)
    pred_texts = [tokenizer.decode(ids) for ids in pred_ids]
    json.dump(pred_texts, open(output_dir / 'pred_texts.json', 'w'))


# model_path = 'google/mt5-base'
args = parse_args()
output_dir = Path(args.output_dir)
MODLE_PATH = 'google/mt5-small'
data_dir = Path(args.data_dir)
os.environ["WANDB_DISABLED"] = "true"

utils.set_seed(0)
utils.dump_args(args, output_dir / 'train_args.json')
print(json.dumps(vars(args), indent=2, ensure_ascii=False))


model = MT5ForConditionalGeneration.from_pretrained(MODLE_PATH)
tokenizer = MT5Tokenizer.from_pretrained(MODLE_PATH)
model = model.cuda()

trainer = get_trainer(model, tokenizer, data_dir, output_dir, args)

clean_test_data = get_dataset(data_dir, 'test', tokenizer=tokenizer)
noisy_test_data = get_dataset(data_dir, 'noisy_test', tokenizer=tokenizer)

predict(trainer, clean_test_data, output_dir / 'test_clean')
predict(trainer, noisy_test_data, output_dir / 'test_noisy')