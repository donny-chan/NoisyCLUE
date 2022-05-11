from pathlib import Path

from transformers import BertTokenizerFast, BertForTokenClassification, BertConfig

from cluener.trainer import CluenerTrainer
from cluener.data import CluenerDataset
from arguments import parse_args
from utils import dump_args


def get_dataset(tokenizer, data_dir, phase, **kwargs):
    # kwargs['num_examples'] = 1024
    return CluenerDataset(tokenizer, data_dir / f'ner_{phase}.json', phase, **kwargs)


def train(trainer: CluenerTrainer, tokenizer, args):
    data_dir = Path(args.data_dir)
    train_dataset = get_dataset(tokenizer, data_dir, 'train')
    dev_dataset = get_dataset(tokenizer, data_dir, 'dev')
    trainer.train(train_dataset, dev_dataset)


def test(trainer, dataset: CluenerDataset, desc: str) -> dict:
    return trainer.evaluate(dataset, desc=desc, output_dir=trainer.output_dir / desc)


def test_all(trainer: CluenerTrainer, tokenizer, args):
    trainer.load_best_ckpt()
    data_dir = Path(args.data_dir)
    for test_phase in ['clean', 'noisy_1', 'noisy_2', 'noisy_3']:
        print(f'testing: {test_phase}', flush=True)
        dataset = get_dataset(tokenizer, data_dir, f'test_{test_phase}')
        test(trainer, dataset, f'test_{test_phase}')


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    dump_args(args, output_dir / 'train_args.json')
    
    print(f'Loading model, path: {args.model_path}', flush=True)
    config = BertConfig.from_pretrained(args.model_path)
    config.num_labels = 32
    model = BertForTokenClassification.from_pretrained(args.model_path, config=config)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_path)
    
    trainer = CluenerTrainer(
        model, 
        output_dir, 
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        grad_acc_steps=args.grad_acc_steps,
        log_interval=args.log_interval,
        lr=args.lr)
    train(trainer, tokenizer, args)
    test_all(trainer, tokenizer, args)



if __name__ == '__main__':
    main()