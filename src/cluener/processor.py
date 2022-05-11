from typing import List, Dict
import json
import copy

from utils import load_jsonl


class Example:
    """A single training/test example for token classification."""
    def __init__(self, guid: str, text: str, labels: List[str]):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class CluenerProcessor:
    def get_label_ids(self, examples: List[Example]) -> List[List[int]]:
        label_to_id = {label: i for i, label in enumerate(self.get_label_list())}
        return [[label_to_id[name] for name in x.labels] for x in examples]

    def tokenize_and_align_labels(self, 
        examples: List[Example], 
        tokenizer, 
        max_length: int) -> Dict[str, List[int]]:
        '''Tokenize and align labels'''
        orig_labels = self.get_label_ids(examples)

        atoms = [list(x.text) for x in examples] # List of lists of atomictokens.
        tokenized_inputs = tokenizer(
            atoms, 
            max_length=max_length, 
            padding='longest', 
            truncation=True, 
            is_split_into_words=True)

        labels = []
        for i, label in enumerate(orig_labels):
            word_ids = tokenized_inputs.word_ids(i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:  # Special tokens
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else: 
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def get_label_list(self) -> List[str]:
        return ["X", "B-address", "B-book", "B-company", 'B-game', 'B-government', 'B-movie', 'B-name',
                'B-organization', 'B-position','B-scene',"I-address",
                "I-book", "I-company", 'I-game', 'I-government', 'I-movie', 'I-name',
                'I-organization', 'I-position','I-scene',
                "S-address", "S-book", "S-company", 'S-game', 'S-government', 'S-movie',
                'S-name', 'S-organization', 'S-position','S-scene',
                'O',"[START]", "[END]"]

    def get_examples(self, file: str, phase: str) -> List[Example]:
        '''
        Read JSON lines from file, convert to BIOS labels.
        '''
        raw_examples = load_jsonl(file)
        examples = []
        for line in raw_examples:
            text = line['text']
            label_entities = line.get('label', None)
            labels = ['O'] * len(text)
            if label_entities is not None:
                # Contruct BIOS labels
                for label_type, entries in label_entities.items():
                    for label_name, pos in entries.items():
                        for raw_lo, raw_hi in pos:
                            for offset in range(-3, 4):  # Check different offsets to acount for wrong labels
                                lo = raw_lo + offset
                                hi = raw_hi + offset
                                if ''.join(text[lo:hi + 1]) != label_name:
                                    # raise ValueError(f"Label name {label_name} does not match text {''.join(text[lo:hi+1])}")
                                    continue
                                if lo == hi:
                                    labels[lo] = f'S-{label_type}'
                                    break
                                else:
                                    labels[lo] = f'B-{label_type}'
                                    labels[lo + 1:hi + 1] = [f'I-{label_type}'] * (len(label_name) - 1)
                                    break
                            else:
                                raise ValueError(f'Label name {label_name} not found in text {"".join(text)}')
            examples.append(Example(
                guid=phase + '-' + line['id'], 
                text=text, 
                labels=labels,
            ))
        return examples

    def convert_examples_to_features(
        self, 
        examples: List[Example], 
        tokenizer,
        max_seq_length: int,
        ) -> Dict[str, List[int]]:
        '''Convert list of examples to list of features'''
        return self.tokenize_and_align_labels(examples, tokenizer, max_seq_length)

