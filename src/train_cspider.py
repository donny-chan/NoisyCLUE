import torch
assert torch.cuda.is_available()

from pathlib import Path
from argparse import Namespace

from transformers import MBartForConditionalGeneration, MBartTokenizerFast

from utils import dump_json, dump_jsonl, get_param_count
from cspider.data import CSpiderDataset
from cspider.trainer import CSpiderTrainer
import arguments


def get_dataset(file, file_schemas, tokenizer, num_examples=None) -> CSpiderDataset:
    return CSpiderDataset(tokenizer, file, file_schemas, num_examples=num_examples)


def test(
    trainer: CSpiderTrainer,
    dataset: CSpiderDataset, 
    output_dir: Path,
    desc: str):
    print('Building dataloader...', flush=True)
    result = trainer.evaluate(dataset, output_dir, desc)

    # Dump results
    print(f'Dumping results to {output_dir}', flush=True)
    output_dir.mkdir(exist_ok=True, parents=True)
    dump_json(result['result'], output_dir / 'result.json')
    dump_jsonl(result['preds'], output_dir / 'preds.json')


def test_all(trainer: CSpiderTrainer, 
             tokenizer: MBartTokenizerFast, 
             data_dir: Path, 
             output_dir: Path,
             args: Namespace):
    print('Testing all...')
    num_opt_steps = None
    trainer.setup_optimizer_and_scheuler(args.lr, num_opt_steps)
    trainer.load_best_ckpt()
    model = trainer.model
    
    # Test Clean
    print('*** Testing clean ***')
    dataset = get_dataset(
        file=data_dir / 'test_clean.json', 
        file_schemas=data_dir / 'schemas.json',
        tokenizer=tokenizer,
        num_examples=args.num_examples)
    test(trainer, dataset, output_dir / 'test_clean', 'test_clean')
    
    # Test Noisy
    for noise_type in ['keyboard', 'asr']:
        for i in range(1, 4):
            test_name = f'test_noisy_{noise_type}_{i}'
            print(f'*** Testing phase: {test_name} ***')
            file_examples = data_dir / f'{test_name}.json'
            print(f'Getting dataset from {file_examples}')
            dataset = get_dataset(
                file_examples, 
                file_schemas=data_dir / 'schemas.json',
                tokenizer=tokenizer,
                num_examples=args.num_examples)
            # Test
            test(trainer, dataset, output_dir / test_name, test_name)


def get_trainer(
    model: MBartForConditionalGeneration, 
    data_dir: Path, 
    output_dir: Path, 
    args: Namespace,
    ) -> CSpiderTrainer:
    print(f'Getting trainer with output dir: {output_dir}')
    return CSpiderTrainer(
        model, 
        output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        grad_acc_steps=args.grad_acc_steps,
        lr=args.lr,
        log_interval=args.log_interval,
    )


def train(
    trainer: CSpiderTrainer, 
    tokenizer: MBartTokenizerFast, 
    data_dir: Path, 
    resume: bool=False,
    args: Namespace=None
    ):
    file_schemas = data_dir / 'schemas.json'
    train_data = CSpiderDataset(
        tokenizer, 
        data_dir / 'train.json', 
        file_schemas=file_schemas,
        num_examples=args.num_examples)
    dev_data = CSpiderDataset(
        tokenizer, data_dir / 'dev.json', 
        file_schemas=file_schemas,
        num_examples=args.num_examples)
    trainer.train(
        train_data, 
        dev_data, 
        resume=resume)


def main():
    args = arguments.parse_args()
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    
    log_file = output_dir / 'train.log'
    # global logger
    # logger = Logger(log_file)

    print('Getting model...', flush=True)
    model = MBartForConditionalGeneration.from_pretrained(args.model_path).cuda()
    print('Getting tokenizer...')
    tokenizer = MBartTokenizerFast.from_pretrained(args.model_path)
    tokenizer.src_lang = "zh_CN"
    tokenizer.tgt_lang = "en_XX"
    print(f'# params: {get_param_count(model)}', flush=True)

    # Train and test
    trainer = get_trainer(model, data_dir, output_dir, args)
    if 'train' in args.mode:
        train(trainer, tokenizer, data_dir, True, args)
    if 'test' in args.mode:
        test_all(trainer, tokenizer, data_dir, output_dir, args)


if __name__ == '__main__':
    main()
