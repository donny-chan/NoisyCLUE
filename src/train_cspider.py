import torch
assert torch.cuda.is_available()

from pathlib import Path

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

from utils import load_jsonl, dump_jsonl, get_param_count, Logger
from cspider.data import CSpiderDataset

logger = None
def log(*args, **kwargs): logger.log(*args, **kwargs)


def get_dataset(tokenizer, file, num_examples=None) -> CSpiderDataset:
    return CSpiderDataset(file, tokenizer, num_examples=num_examples)


def test(model, dataset, output_dir: Path):
    log('Building dataloader...')
    dataloader = DataLoader(dataset, batch_size=16)

    # Results
    preds = []
    
    log('*** Testing ***')
    log(f'# steps: {len(dataloader)}')
    log(f'# examples: {len(dataset)}')
    for batch in tqdm(dataloader):
        for k, v in batch.items(): batch[k] = v.cuda()  # Move to GPU
        generated_tokens = model.generate(**batch)
        output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)  # Will move to CPU automatically
        preds += output
    log('*** Done testing ***')

    # Dump results
    log(f'Dumping results')
    output_dir.mkdir(exist_ok=True, parents=True)
    dump_jsonl(preds, output_dir / 'preds.json')


def test_all(model, tokenizer, data_dir: Path, output_dir: Path):
    log('Testing all...')
    for phase in ['clean', 'noisy_1', 'noisy_2', 'noisy_3']:
        logger.log(phase)
        examples_file = data_dir / f'nmt_test_{phase}.json'
        logger.log(f'Getting dataset from {examples_file}')
        dataset = get_dataset(tokenizer, examples_file)
        # Test
        test(model, dataset, output_dir / f'test_{phase}')


def get_trainer(model, data_dir: Path, output_dir: Path) -> Trainer:
    


def train(model, tokenizer, data_dir: Path, output_dir: Path):
    trainer 


model_path = "facebook/mbart-large-50-many-to-one-mmt"

output_dir = Path('results/cspider/mbart-large')
data_dir = Path('../data/keyboard/cspider')
log_file = output_dir / 'train.log'

logger = Logger(log_file)

log('Getting model...')
model = MBartForConditionalGeneration.from_pretrained(model_path).cuda()
log('Getting tokenizer...')
tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
tokenizer.src_lang = "zh_CN"
log(f'# params: {get_param_count(model)}')


if __name__ == '__main__':
    train(model, tokenizer, data_dir, output_dir)
    test_all(model, tokenizer, data_dir, output_dir)
