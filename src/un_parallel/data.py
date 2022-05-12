from pathlib import Path
import json
import random

from torch import LongTensor
from torch.utils.data import Dataset, IterableDataset


class UNParallelExample:
    def __init__(self, en, zh):
        self.en = en
        self.zh = zh

    def __repr__(self):
        return self.en + ' <---> ' + self.zh


class UNParallelEnZhDataset(Dataset):
    def __init__(self, tokenizer, en_lines: list, zh_lines: list, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.en_lines = en_lines
        self.zh_lines = zh_lines
        # self.features = self.get_features(en_lines, zh_lines)

    def get_features(self, en_lines: list, zh_lines: list) -> list:
        print(f'Building features from {len(en_lines)} pairs of sentences...', flush=True)
        print('Tokenizing Chinese sentences', flush=True)
        features = self.tokenizer(zh_lines, max_length=self.max_len, truncation=True, padding=True)
        print('Tokenizing English sentences', flush=True)
        labels = self.tokenizer(en_lines, max_length=self.max_len, truncation=True, padding=True)
        features['labels'] = labels['input_ids']
        return features

    def __len__(self) -> int:
        # return len(self.features['input_ids'])
        return len(self.zh_lines)

    def __getitem__(self, idx: int) -> dict:
        # return {
        #     key: self.features[key][idx] for key in ['input_ids', 'attention_mask', 'labels']
        # }
        kwargs = {
            'max_length': self.max_len,
            'truncation': True,
            'padding': 'max_length',
            'return_tensors': 'pt',
        }
        inputs = self.tokenizer(self.zh_lines[idx], **kwargs)
        labels = self.tokenizer(self.en_lines[idx], **kwargs)

        for k in inputs:
            inputs[k] = inputs[k].squeeze()
        inputs['labels'] = labels['input_ids'].squeeze()
        return inputs


class UNParallelZhEnIterableDataset(IterableDataset):
    def __init__(self, features_file: Path, cache_size: int, num_examples: int):
        super().__init__()
        self.cache_size = cache_size
        self.features_file = Path(features_file)
        self.num_examples = num_examples

        self.reset()

    def chunked_iter(self):
        '''
        Iterate over features file in chunks of size `self.cache_size`.
        '''
        chunk = []
        for line in self.reader:
            chunk.append(json.loads(line))
            if len(chunk) == self.cache_size:
                yield chunk
                chunk = []
        if chunk != []:
            yield chunk

    def reset(self):
        self.count = 0
        self.reader = open(self.features_file, 'r', encoding='utf8')

    def __len__(self):
        return self.num_examples

    def __iter__(self):
        for chunk in self.chunked_iter():
            indices = list(range(len(chunk)))
            random.shuffle(indices)
            # print('indices:', indices)
            # for sample in chunk:
            #     print(sample['input_ids'][:10])
            for i in indices:
                if self.count == self.num_examples:
                    break
                for k, v in chunk[i].items():
                    chunk[i][k] = LongTensor(v)
                yield chunk[i]
                self.count += 1
            if self.count == self.num_examples:
                break
        self.reader.close()