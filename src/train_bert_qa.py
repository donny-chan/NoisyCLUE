# coding: utf8
import torch
assert torch.cuda.is_available()

from pathlib import Path

from transformers import BertForQuestionAnswering, BertTokenizer
from cmrc2018 import CMRC2018Dataset, Trainer
import utils
import arguments


def test(trainer: Trainer, dataset: CMRC2018Dataset, output_dir: Path, desc: str):
    '''
    Test model on given dataset
    '''
    eval_output = trainer.evaluate(dataset, output_dir, is_labeled=True, desc=f'Testing {desc}')

    # Save results
    result = eval_output['result']
    preds = eval_output['preds']
    print(result, flush=True)
    utils.dump_json(result, output_dir / 'result.json')
    utils.dump_json(preds, output_dir / 'preds.json')


def test_all(trainer: Trainer, data_dir: Path, output_dir: Path, tok_name: str):
    print('test_all', flush=True)
    trainer.load_best_model(output_dir)
    for test_type in ['clean', 'noisy_1', 'noisy_2', 'noisy_3']:
        examples_file = data_dir / f'cmrc2018_test_{test_type}.json'
        dataset = CMRC2018Dataset(trainer.tokenizer, examples_file, 
            has_labels=True, tok_name=tok_name)
        test(trainer, dataset, output_dir=output_dir / f'test_{test_type}', 
             desc=test_type)


def main():
    args = arguments.parse_args()
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Model
    print('Loading model', flush=True)
    model = BertForQuestionAnswering.from_pretrained(args.model_path)
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    tok_name = args.model_path.split('/')[-1]

    print(f'# params: {utils.get_param_count(model)}', flush=True)

    utils.set_seed(0)
    trainer = Trainer(model, tokenizer, args)
    if not args.resume_from_checkpoint:
        # Data
        print('Loading train and dev data', flush=True)
        train_dataset = CMRC2018Dataset(tokenizer, data_dir / 'cmrc2018_train.json', 
            has_labels=True, tok_name=tok_name)
        eval_dataset = CMRC2018Dataset(tokenizer, data_dir / 'cmrc2018_dev.json', 
            has_labels=True, tok_name=tok_name)
        trainer.train(train_dataset, eval_dataset)
    test_all(trainer, data_dir, output_dir, tok_name)


if __name__ == '__main__':
    # from multiprocessing import freeze_support
    # freeze_support()
    main()