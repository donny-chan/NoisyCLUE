import torch
assert torch.cuda.is_available()

from pathlib import Path

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from utils import dump_jsonl, get_param_count, Logger
from nmt.data import NmtDataset


logger = None
def log(*args, **kwargs): logger.log(*args, **kwargs)


def get_dataset(tokenizer, file, num_examples=None):
    return NmtDataset(file, tokenizer, num_examples=num_examples)


def test(model, tokenizer, dataset, output_dir: Path, batch_size=16):
    log('Building dataloader...')
    dataloader = DataLoader(dataset, batch_size=batch_size)
    lang_id_en = tokenizer.get_lang_id('en')

    # Results
    preds = []
    
    log('*** Testing ***')
    log(f'Batch size: {batch_size}')
    log(f'# steps: {len(dataloader)}')
    log(f'# examples: {len(dataset)}')
    for batch in tqdm(dataloader):
        for k, v in batch.items(): batch[k] = v.cuda()  # Move to GPU
        generated_tokens = model.generate(**batch, forced_bos_token_id=lang_id_en)
        output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)  # Will move to CPU automatically
        preds += output
    log('*** Done testing ***')

    # Dump results
    output_dir.mkdir(exist_ok=True, parents=True)
    log(f'Dumping result to {output_dir}')
    dump_jsonl(preds, output_dir / 'preds.json')


def test_all(model, tokenizer, data_dir: Path, output_dir: Path):
    # Test clean 
    def test_clean():
        file_examples = data_dir / 'test_clean.json'
        log(f'*** Testing phase: clean ***')
        log(f'Loading from {file_examples}')
        dataset = get_dataset(tokenizer, file_examples)
        test(model, tokenizer, dataset, output_dir / 'test_clean')
    
    def test_noisy():
        # Test noisy
        for noise_type in ['keyboard', 'asr']:
            for i in range(2, 4):
                test_name = f'test_noisy_{noise_type}_{i}'
                log(f'*** Testing phase: {test_name} ***')
                file_examples = data_dir / f'{test_name}.json'
                dataset = get_dataset(tokenizer, file_examples)
                test(model, tokenizer, dataset, output_dir / test_name)
                del dataset

    model.eval()
    # test_clean()
    test_noisy()


if __name__ == '__main__':
    model_name = "m2m100_418M"
    model_path = f"facebook/{model_name}"

    output_dir = Path(f'results/nmt/{model_name}')
    data_dir = Path('../data/realtypo/nmt')
    log_file = output_dir / 'test.log'

    logger = Logger(log_file)

    log('Getting model...')
    model = M2M100ForConditionalGeneration.from_pretrained(model_path).cuda()
    log('Getting tokenizer...')
    tokenizer = M2M100Tokenizer.from_pretrained(model_path)
    tokenizer.src_lang = "zh"
    log(f'# params: {get_param_count(model)}')
    
    # Test
    test_all(model, tokenizer, data_dir, output_dir)
