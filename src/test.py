from transformers import AutoTokenizer

from utils import iter_jsonl

tokenizer = AutoTokenizer.from_pretrained('google/mt5-base')
dev_file = 'un_parallel/dev.json'


i = 0
for sample in iter_jsonl(dev_file):
    input_ids = tokenizer.convert_ids_to_tokens(sample['input_ids'])
    labels = tokenizer.convert_ids_to_tokens(sample['labels'])
    print(input_ids[:10])
    print(labels[:10])
    i += 1
    if i == 5:
        exit()
