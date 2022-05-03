import torch
assert torch.cuda.is_available()

import time
from pathlib import Path
from argparse import Namespace

from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from data.afqmc import AfqmcSeq2SeqDataset
import utils
import utils_seq2seq
from arguments import parse_args


def get_trainer(
    model: MT5ForConditionalGeneration, tokenizer: MT5Tokenizer, data_dir: Path,
    output_dir: Path, args: Namespace
) -> Seq2SeqTrainer:
    '''Return a huggingface Trainer instance.'''
    kwargs = {'tokenizer': tokenizer, 'num_examples': args.num_examples}
    train_dataset = utils_seq2seq.get_dataset(data_dir, 'train', **kwargs)
    eval_dataset = utils_seq2seq.get_dataset(data_dir, 'dev', **kwargs)
    print('# train examples:', len(train_dataset))
    print('# eval examples:', len(eval_dataset))

    # Hyperparameters
    batch_size = args.batch_size
    grad_acc_steps = args.grad_acc_steps
    num_epochs = args.num_epochs
    warmup_ratio = 0.1
    lr = args.lr
    
    train_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True, # TODO: remove on release
        do_train=True,
        do_predict=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc_steps,
        # Move predictions to CPU often because vocab is very large.
        eval_accumulation_steps=128,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=lr,
        num_train_epochs=num_epochs,
        lr_scheduler_type='linear',
        optim='adafactor',
        warmup_ratio=warmup_ratio,
        report_to='none',
        logging_first_step=True,
        logging_steps=args.log_interval,
        disable_tqdm=not args.tqdm,
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
    )
    trainer = Seq2SeqTrainer(
        model,
        train_args,
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset,
    )
    return trainer


def train(trainer: Seq2SeqTrainer, args: Namespace):
    train_args = {}
    if args.resume_from_checkpoint:
        train_args['resume_from_checkpoint'] = True
    start_time = time.time()
    trainer.train(**train_args)
    print('Training time:', time.time() - start_time)


def predict(trainer: Seq2SeqTrainer, dataset: AfqmcSeq2SeqDataset,
            output_dir: Path, args: Namespace):
    preds, result = utils_seq2seq.predict(trainer, dataset, output_dir, args)
    print(result)

    # Save
    output_dir.mkdir(exist_ok=True, parents=True)
    utils.dump_json(result, output_dir / 'result.json')
    preds_text = [tokenizer.decode(ids) for ids in preds]  # (N, ), list of str
    utils.dump_str(preds, output_dir / 'preds.txt')
    utils.dump_json(preds_text, output_dir / 'preds_text.json', indent=2)


def test(trainer: Seq2SeqTrainer, tokenizer, data_dir: Path):
    for test_phase in ['test_clean', 'test_noisy_1', 'test_noisy_2', 'test_noisy_3']:
        start_time = time.time()
        print('\nTesting phase:', test_phase)
        data = utils_seq2seq.get_dataset(data_dir, test_phase, tokenizer=tokenizer)
        predict(trainer, data, output_dir / test_phase, args)
        print('test_run_time:', time.time() - start_time)


args = parse_args()
output_dir = Path(args.output_dir)
data_dir = Path(args.data_dir)

utils.set_seed(args.seed)
utils.dump_args(args, output_dir / 'train_args.json')

# Get model
model = MT5ForConditionalGeneration.from_pretrained(args.model_path)
tokenizer = MT5Tokenizer.from_pretrained(args.model_path)
print('# parameters:', utils.get_param_count(model))

# Train and test
trainer = get_trainer(model, tokenizer, data_dir, output_dir, args)
train(trainer, args)
test(trainer, tokenizer, data_dir)