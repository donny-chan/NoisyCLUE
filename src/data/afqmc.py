import os.path as osp

import torch
from torch.utils.data import Dataset

from . import utils


def get_examples(file: str, set_type: str) -> list:
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(utils.iter_jsonl(file)):
        guid = "%s-%s" % (set_type, i)
        text_a = line['sentence1']
        text_b = line['sentence2']
        # label = str(line['label']) if set_type != 'test' else "0"
        label = str(line['label'])
        examples.append({
            'guid': guid, 
            'text': [text_a, text_b],
            'label': label,
        })
    return examples

def get_train_examples(data_dir):
    return get_examples(osp.join(data_dir, "train.json"), "train")
def get_dev_examples(data_dir):
    return get_examples(osp.join(data_dir, "dev.json"), "dev")
def get_test_examples(data_dir):
    return get_examples(osp.join(data_dir, "test.json"), "test")
def get_noisy_test_examples(data_dir):
    return get_examples(osp.join(data_dir, "noisy_test.json"), "noisy_test")


class AfqmcDataset(Dataset):
    def __init__(self, file: str, phase: str, tokenizer, max_seq_len: int, 
                 num_examples: int=None):
        self.file = file
        # self.examples = get_examples(file, phase)
        examples = get_examples(file, phase)[:num_examples]
        self.features = self.get_features(examples, tokenizer, max_seq_len)

    def get_features(self, examples: list, tokenizer, max_seq_len: int) -> dict:
        """
        Return list of examples (dict) into dict of features:
        ```
        {
            'input_ids': [...],
            'token_type_ids': [...],
            'attention_mask': [...],
            'labels': [...],
        }
        ```
        """
        label_list = ['0', '1']
        label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        texts = [x['text'] for x in examples]
        features = tokenizer(
            texts,
            max_length=max_seq_len,
            truncation='longest_first',
            padding='max_length',
            return_tensors='pt')
        features['labels'] = torch.tensor([label_map[x['label']] for x in examples])
        return features

    def __getitem__(self, idx):
        return {
            k: self.features[k][idx] for k in 
            ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
        }
    
    def __len__(self):
        return len(self.features['input_ids'])

class AfqmcSeq2SeqDataset(Dataset):
    def __init__(self, file: str, phase: str, tokenizer, num_examples: int=None):
        self.verbalizer = ['non_equivalent', 'equivalent']
        self.file = file
        self.tokenizer = tokenizer

        examples = get_examples(file, phase)[:num_examples]
        self.features = self.get_features(examples, tokenizer)

    def get_features(self, examples: list, tokenizer) -> dict:
        '''
        A feature for seq2seq is a pair of input_ids and labels.

        input text template:  "afqmc。句子1：{}，句子2：{}。"
        output text template: "{}"

        Return:
        ```
        {
            'input_ids': [...],
            'labels': [...]
        }
        ```
        '''
        source_template = 'afqmc。句子1：{}，句子2：{}。'
        texts = [source_template.format(ex['text'][0], ex['text'][1]) for ex in examples]
        label_ids = [int(ex['label']) for ex in examples]
        labels = [self.verbalizer[label_id] for label_id in label_ids]
        input_ids = tokenizer(texts, padding=True).input_ids
        labels = tokenizer(labels, padding=True).input_ids
        return {'input_ids': input_ids, 'labels': labels}

    def __getitem__(self, idx):
        return {
            k: self.features[k][idx] for k in 
            ['input_ids', 'labels']
        }
    def __len__(self):
        return len(self.features['input_ids'])
