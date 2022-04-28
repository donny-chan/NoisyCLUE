import json
import os.path as osp

from torch.utils.data import Dataset

def iter_jsonl(file):
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            yield json.loads(line.strip())

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

def get_features(examples, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    verbalizer = {'0': 'non_equivalent', '1': 'equivalent'}
    
    source_template = 'afqmc。句子1：{}，句子2：{}。'
    target_template = '{}'
    
    features = {'input_ids': [], 'labels': []}
    texts = [source_template.format(ex['text'][0], ex['text'][1]) for ex in examples]
    labels = [verbalizer[ex['label']] for ex in examples]
    input_ids = tokenizer(texts, padding=True).input_ids
    labels = tokenizer(labels, padding=True).input_ids
    return {'input_ids': input_ids, 'labels': labels}
    # for eg in examples:
    #     # Construct input and label
    #     texts = eg['text']
    #     label = verbalizer[eg['label']]
    #     source_text = source_template.format(texts[0], texts[1])
    #     target_text = target_template.format(label)
        
    #     # Tokenize
    #     input_ids = tokenizer(source_text, return_tensors='pt').input_ids
    #     labels = tokenizer(target_text, return_tensors='pt').input_ids
    #     features['input_ids'].append(input_ids)
    #     features['labels'].append(labels)
    # return features

class AfqmcDataset(Dataset):
    def __init__(self, file: str, phase: str, tokenizer, num_examples: int=None):
        self.file = file
        examples = get_examples(file, phase)[:num_examples]
        self.features = get_features(examples, tokenizer)
    def __getitem__(self, idx):
        return {
            k: self.features[k][idx] for k in 
            ['input_ids', 'labels']
        }
    def __len__(self):
        return len(self.features['input_ids'])
