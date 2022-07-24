from torch.utils.data import Dataset

from utils import load_jsonl, load_json


class CSpiderDataset(Dataset):
    def __init__(self, tokenizer, examples_file, file_schemas, max_len=512, num_examples=None):
        self.tokenizer = tokenizer
        self.examples_file = examples_file
        self.max_len = max_len
        self.schemas =  load_json(file_schemas)
        self.examples = self.get_examples(examples_file)[:num_examples]
        self.features = self.get_features(self.examples)

    def get_examples(self, file):
        print(f'Loading examples from {file}')
        examples = load_jsonl(file)
        return examples

    def get_features(self, examples):
        questions = [e['question'] for e in examples]
        queries = [e['query'] for e in examples]
        kwargs = {
            'padding': True,
            'truncation': True,
            'max_length': self.max_len,
            'return_tensors': 'pt',
        }
        
        input_texts = []
        for i, question in enumerate(questions):
            db_id = examples[i]['db_id']
            schema = self.schemas[db_id]  # dict (table -> columns)
            schema_text = ''
            for table, columns in schema.items():
                schema_text += table + ' ( ' + ' '.join(columns) + ' ) '
            schema_text = schema_text[:-1]
            # Truncate such that schema + '：' + question fits in max_len
            if len(schema_text) + 1 + len(question) > self.max_len:
                schema_text = schema_text[:self.max_len - len(question) - 1]
            input_text = schema_text + '：' + question
            if i == 0:
                print('*** Example ***')
                print('Input text:')
                print(input_text)
                print('query:')
                print(queries[i])
                print('***************')
                # exit()
            input_texts.append(input_text)
        
        inputs = self.tokenizer(input_texts, **kwargs)
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
