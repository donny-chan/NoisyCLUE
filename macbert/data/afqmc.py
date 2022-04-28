import json
import os.path as osp

from torch.utils.data import Dataset

from . import utils


def iter_jsonl(file):
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            yield json.loads(line.strip())


def load_jsonl(file):
    examples = []
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            line = json.loads(line.strip())
            examples.append(line)
    return examples

def get_examples(file, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(iter_jsonl(file)):
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
        self.features = utils.get_features(examples, self.get_labels(), max_seq_len, tokenizer)

    def get_labels(self):
        return ['0', '1']

    def __getitem__(self, idx):
        return {
            k: self.features[k][idx] for k in 
            ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
        }
    
    def __len__(self):
        return len(self.features['input_ids'])
