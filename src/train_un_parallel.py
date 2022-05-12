from pathlib import Path

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
)
from transformers.optimization import Adafactor, AdafactorSchedule

from un_parallel.trainer import UNParallelTrainer
from un_parallel.data import UNParallelEnZhDataset, UNParallelZhEnIterableDataset
from arguments import parse_args
from utils import dump_args

args = parse_args()
output_dir = Path(args.output_dir)
log_file = output_dir / 'train.log'
log_writer = open(log_file, 'w', encoding='utf8')
dump_args(args, output_dir / 'train_args.json')

def log(*args, **kwargs):
    print(*args, **kwargs, flush=True)
    print(*args, **kwargs, file=log_writer, flush=True)


def get_datasets(tokenizer, data_dir):
    def get_lines(data_dir: Path, prefix='UNv1.0.en-zh') -> tuple:
        log(f'Loading examples from "{data_dir}", prefix: "{prefix}"')
        log('Loading English sentences...')
        with open(data_dir / f'{prefix}.en', 'r') as f:
            en_lines = [line.strip() for line in tqdm(f, mininterval=2)]
        log(f'Loading Chinese sentences...')
        with open(data_dir / f'{prefix}.zh', 'r') as f:
            zh_lines = [line.strip() for line in tqdm(f, mininterval=2)]
        log(f'Loaded {len(en_lines)} pairs of sentences')
        return en_lines, zh_lines
    log('Loading sentences...')
    en_lines, zh_lines = get_lines(data_dir, 'small')
    # en_lines, zh_lines = get_lines(data_dir)
    log('Splitting list of sentences...')
    train_en_lines = en_lines[:int(len(en_lines) * 0.9)]
    train_zh_lines = zh_lines[:int(len(zh_lines) * 0.9)]
    dev_en_lines = en_lines[int(len(en_lines) * 0.9):]
    dev_zh_lines = zh_lines[int(len(zh_lines) * 0.9):]
    del en_lines, zh_lines

    log('Building datasets...')
    train_dataset = UNParallelEnZhDataset(tokenizer, train_en_lines, train_zh_lines, max_len=128)
    dev_dataset = UNParallelEnZhDataset(tokenizer, dev_en_lines, dev_zh_lines, max_len=128)
    return train_dataset, dev_dataset


def get_iterable_datasets():
    train_dataset = UNParallelZhEnIterableDataset('un_parallel/features.json', cache_size=2**10, num_examples=15886041)
    dev_dataset = UNParallelZhEnIterableDataset('un_parallel/dev.json', cache_size=2**10, num_examples=100)
    return train_dataset, dev_dataset


def get_trainer(model, tokenizer, data_dir, output_dir, args):
    # log('Building datasets...')
    # train_dataset, dev_dataset = get_datasets(tokenizer, en_lines, zh_lines)

    log('Building datasets...')
    train_dataset, dev_dataset = get_iterable_datasets()

    log('Building trainer...')
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy='steps',
        eval_steps=100,
        save_strategy='steps',
        save_steps=500,
        max_steps=100000,
        report_to='none',
        logging_steps=args.log_interval,
        optim='adafactor',
        lr_scheduler_type='constant',
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc_steps,
        weight_decay=0.01,
        # warmup_ratio=0.1,
        disable_tqdm=True,
        num_train_epochs=args.num_epochs,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        # tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
    )
    return trainer


def train(model, output_dir, args):
    log('Building iterable datasets...')
    train_dataset, dev_dataset = get_iterable_datasets()

    trainer = UNParallelTrainer(
        model,
        output_dir,
        lr=args.lr,
        grad_acc_steps=args.grad_acc_steps,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
        num_epochs=args.num_epochs,
        eval_interval=1024,
        eval_strategy='step',
    )

    trainer.train(train_dataset, dev_dataset)




data_dir = Path('../data/un-parallel/en-zh')

# log('Building tokenizer...')
# tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer = None
log('Building model...')
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)

# trainer = get_trainer(model, tokenizer, data_dir, output_dir, args)
# trainer.train()

train(model, output_dir, args)

