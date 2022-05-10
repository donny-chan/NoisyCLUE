from pathlib import Path
from typing import List

from torch import LongTensor
from torch.utils.data import Dataset

from .processor import CluenerProcessor, Example
from utils import load_jsonl


def get_examples(examples_file: Path, num_examples: int=None) -> List[Example]:
    raw_data = load_jsonl(examples_file, num_examples=num_examples)
    examples: List[Example] = []
    for line in raw_data:
        examples.append(Example(
            guid=line['id'],
            text=line['text'],
            labels=line['label'],
        ))
    return examples


class CluenerDataset(Dataset):
    def __init__(self, tokenizer, examples_file: str, phase: str, num_examples: int=None):
        self.tokenizer = tokenizer
        self.examples_file = Path(examples_file)

        # Process data
        self.processor = CluenerProcessor()
        self.examples = self.processor.get_examples(examples_file, phase)
        self.examples = self.examples[:num_examples]
        self.features = self.get_features(self.examples)

    def get_id2label(self) -> List[str]:
        return self.processor.get_label_list()

    def get_features(self, examples) -> List[dict]:
        return self.processor.convert_examples_to_features(
            examples, self.tokenizer, 512)

    def __getitem__(self, idx: int) -> dict:
        keys = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
        return {key: LongTensor(self.features[key][idx]) for key in keys}
    
    def __len__(self) -> int:
        return len(self.features['input_ids'])
