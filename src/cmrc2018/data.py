from pathlib import Path
from typing import Tuple, List

import torch
from torch.utils.data import Dataset

from .processor import (
    read_squad_examples, 
    convert_examples_to_features, 
    SquadExample,
)
from utils import dump_jsonl, load_jsonl


def extract_data(features):
    data = {
        'input_ids': [],
        'input_mask': [],
        # 'span_mask': [],
        'segment_ids': [],
        'start': [],
        'end': []
    }
    for feature in features:
        data['input_ids'].append(feature.input_ids)
        data['input_mask'].append(feature.input_mask)
        # data['span_mask'].append(feature.input_span_mask)
        data['segment_ids'].append(feature.segment_ids)
        data['start'].append(feature.start_position)
        data['end'].append(feature.end_position)
    return data


def get_data(
    tokenizer, 
    examples_file: Path, 
    has_labels: bool, 
    tok_name: str,
    num_examples: int=None,  # TODO: Change to None on release
    use_cache: bool=True,
    ) -> Tuple[List[SquadExample], List[dict]]:
    '''
    Get examples, convert to features.
    return: `(examples, features)`
    '''
    def log(*args, **kwargs): 
        print(*args, **kwargs, flush=True)

    def load_features_cache(file: Path) -> List[dict]:
        features = load_jsonl(file)
        # Convert str keys to int
        for i, f in enumerate(features):
            for key in ['token_to_orig_map', 'token_is_max_context']:
                features[i][key] = {int(k): v for k, v in f[key].items()}
        return features

    data_dir = examples_file.parent
    log('Loading examples...')
    examples = read_squad_examples(
        examples_file, has_labels=has_labels, num_examples=num_examples)
    log(f'Loaded {len(examples)} examples.')

    # Get features (load cache or convert from examples)
    if num_examples:
        tok_name = f'{num_examples}_{tok_name}'
    features_cache = data_dir / '.cache' / f'features_{tok_name}_{examples_file.name}'
    features_cache.parent.mkdir(exist_ok=True, parents=True)

    if use_cache and features_cache.exists():
        log('Loading features from cache...')
        features = load_features_cache(features_cache)
    else:
        log(f'Converting {len(examples)} examples to features')
        features = convert_examples_to_features(
            examples, tokenizer, has_labels=has_labels)
        dump_jsonl(features, features_cache)
    log(f'Got {len(features)} features')
    return examples, features


class CMRC2018Dataset(Dataset):
    '''
    CMRC2018 dataset
    '''
    def __init__(self, tokenizer, examples_file: str, has_labels: bool, tok_name: str):
        super().__init__()
        self.file = Path(examples_file)
        self.has_labels = has_labels
        self.examples, self.features = get_data(
            tokenizer, self.file, has_labels=has_labels, tok_name=tok_name)
    
    def __len__(self):
        # return len(self.data['input_ids'])
        return len(self.features)

    def __getitem__(self, idx: int):
        feat = self.features[idx]
        # input_ids = torch.LongTensor(feat['input_ids'])
        # input_mask = torch.LongTensor(feat['input_mask'])
        # segment_ids = torch.LongTensor(feat['segment_ids'])
        keys = ['input_ids', 'attention_mask', 'segment_ids']
        ret = {key: torch.LongTensor(feat[key]) for key in keys}
        if self.has_labels:
            ret['start_position'] = torch.LongTensor([feat['start_position']])
            ret['end_position'] = torch.LongTensor([feat['end_position']])
        ret['idx'] = torch.LongTensor([idx])
        return ret

    def get_query_id_to_answers(self):
        '''Return: {query_id: [ans0, ans1]}'''
        examples = load_jsonl(self.file)
        ret = {}
        for eg in examples:
            ret[eg['id']] = [x['text'] for x in eg['answers']]
        return ret
