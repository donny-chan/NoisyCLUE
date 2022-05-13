import torch
assert torch.cuda.is_available()

from pathlib import Path

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

from utils import load_jsonl, dump_jsonl, get_param_count, Logger
from nmt.data import NmtDataset


logger = None
def log(*args, **kwargs): logger.log(*args, **kwargs)


def get_dataset(tokenizer, file):
    return NmtDataset(file, tokenizer, 128)


def test(model, dataset, output_dir: Path):
    log('Building dataloader...')
    dataloader = DataLoader(dataset, batch_size=8)

    # Results
    preds = []
    
    log('*** Testing ***')
    for batch in tqdm(dataloader):
        for k, v in batch.items(): batch[k] = v.cuda()  # Move to GPU
        generated_tokens = model.generate(**batch)
        output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)  # Will move to CPU automatically
        preds += output
    log('*** Done testing ***')

    # Dump results
    log(f'Dumping results')
    dump_jsonl(preds, output_dir / 'preds.json')


def test_all(model, tokenizer, data_dir: Path, output_dir: Path):
    log('test_all')
    for phase in ['clean', 'noisy_1', 'noisy_2', 'noisy_3']:
        logger.log(phase)
        examples_file = data_dir / f'nmt_test_{phase}.json'
        logger.log(f'Getting dataset from {examples_file}')
        dataset = get_dataset(tokenizer, examples_file)
        test(model, dataset, output_dir / phase)


model_path = "facebook/mbart-large-50-many-to-one-mmt"

output_dir = Path('results/nmt/mbart-large')
data_dir = Path('../data/keyboard/nmt')
log_file = output_dir / 'test.log'

logger = Logger(log_file)

log('Getting model...')
model = MBartForConditionalGeneration.from_pretrained(model_path).cuda()
log('Getting tokenizer...')
tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
tokenizer.src_lang = "zh_CN"
log(f'# params: {get_param_count(model)}')


if __name__ == '__main__':
    test_all(model, tokenizer, data_dir, output_dir)
