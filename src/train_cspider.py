import torch
assert torch.cuda.is_available()

from pathlib import Path
from argparse import Namespace

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import MBartForConditionalGeneration, MBartTokenizerFast

from utils import dump_jsonl, get_param_count, Logger
from cspider.data import CSpiderDataset
from cspider.trainer import CSpiderTrainer
import arguments

logger = None
def log(*args, **kwargs): logger.log(*args, **kwargs)


def get_dataset(file, tokenizer, num_examples=None) -> CSpiderDataset:
    return CSpiderDataset(tokenizer, file, num_examples=num_examples)


def test(model: MBartForConditionalGeneration,
         tokenizer: MBartTokenizerFast, 
         dataset: CSpiderDataset, 
         output_dir: Path):
    log('Building dataloader...')
    dataloader = DataLoader(dataset, batch_size=4)

    # Results
    preds = []
    
    log('*** Testing ***')
    log(f'# steps: {len(dataloader)}')
    log(f'# examples: {len(dataset)}')
    for batch in tqdm(dataloader):
        for k, v in batch.items(): batch[k] = v.cuda()  # Move to GPU
        generated_tokens = model.generate(
            input_ids=batch['input_ids'].cuda(),
            attention_mask=batch['attention_mask'].cuda(),
        )
        output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)  # Will move to CPU automatically
        preds += output
    log('*** Done testing ***')

    # Dump results
    log(f'Dumping results')
    output_dir.mkdir(exist_ok=True, parents=True)
    dump_jsonl(preds, output_dir / 'preds.json')


def test_all(trainer: CSpiderTrainer, 
             tokenizer: MBartTokenizerFast, 
             data_dir: Path, 
             output_dir: Path,
             args: Namespace):
    log('Testing all...')
    trainer.setup_optimizer_and_scheuler(args.lr, 0)
    trainer.load_best_ckpt()
    model = trainer.model
    
    # Test Clean
    log('*** Testing clean ***')
    dataset = get_dataset(
        file=data_dir / 'test_clean.json', 
        tokenizer=tokenizer,
        num_examples=args.num_examples)
    test(model, tokenizer, dataset, output_dir / 'test_clean')
    
    # Test Noisy
    for noise_type in ['keyboard', 'asr']:
        for i in range(1, 4):
            test_name = f'test_noisy_{noise_type}_{i}'
            log(f'*** Testing phase: {test_name} ***')
            file_examples = data_dir / f'{test_name}.json'
            log(f'Getting dataset from {file_examples}')
            dataset = get_dataset(file_examples, tokenizer,
                                  num_examples=args.num_examples)
            # Test
            test(model, tokenizer, dataset, output_dir / test_name)


def get_trainer(model, data_dir: Path, output_dir: Path, args: Namespace) -> CSpiderTrainer:
    log(f'Getting trainer with output dir: {output_dir}')
    return CSpiderTrainer(
        model, 
        output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        grad_acc_steps=args.grad_acc_steps,
        lr=args.lr,
        log_interval=args.log_interval,
    )


def train(trainer: CSpiderTrainer, 
          tokenizer: MBartTokenizerFast, 
          data_dir: Path, 
          output_dir: Path, 
          args: Namespace=None):
    train_data = CSpiderDataset(tokenizer, data_dir / 'train.json', num_examples=args.num_examples)
    dev_data = CSpiderDataset(tokenizer, data_dir / 'dev.json', num_examples=args.num_examples)
    
    trainer.train(
        train_data, 
        dev_data, 
        resume=False)


def main():
    args = arguments.parse_args()
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    log_file = output_dir / 'train.log'
    global logger
    logger = Logger(log_file)

    log('Getting model...')
    model = MBartForConditionalGeneration.from_pretrained(args.model_path).cuda()
    log('Getting tokenizer...')
    tokenizer = MBartTokenizerFast.from_pretrained(args.model_path)
    tokenizer.src_lang = "zh_CN"
    tokenizer.tgt_lang = "en_XX"
    log(f'# params: {get_param_count(model)}')

    # Train and test
    trainer = get_trainer(model, data_dir, output_dir, args)
    train(trainer, tokenizer, data_dir, output_dir, args)
    test_all(trainer, tokenizer, data_dir, output_dir)


if __name__ == '__main__':
    main()
