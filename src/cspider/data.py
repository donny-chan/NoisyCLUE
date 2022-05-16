from torch.utils.data import Dataset

from utils import load_jsonl


class CSpiderDataset(Dataset):
    def __init__(self, tokenizer, examples_file):
        self.tokenizer = tokenizer
        self.examples_file = examples_file
        self.examples = self.get_examples(examples_file)
        

    def get_examples(self, file):
        return load_jsonl(file)

    def get_features(self, examples):
        questions = [e['question'] for e in examples]
        queries = [e['query'] for e in examples]
        kwargs = {
            'padding': True,
            'truncation': True,
            'max_length': self.max_len,
            'return_tensors': 'pt',
        }
        inputs = self.tokenizer(questions, **kwargs)
        labels = self.tokenizer(queries, **kwargs)
        features = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids'],
        }
        return features
    
    def __len__(self) -> int:
        return len(self.features['input_ids'])

    def __getitem__(self, idx: int) -> dict:
        return {key: self.features[key][idx] for key in self.features}
