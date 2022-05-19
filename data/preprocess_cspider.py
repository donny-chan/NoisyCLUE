from pathlib import Path
from utils import load_jsonl, dump_jsonl


def process_file(file, question_key):
    examples = load_jsonl(file)
    parsed_examples = []
    for example in examples:
        parsed_examples.append({
            'question': example[question_key],
            'query': example['query'],
        })
    return parsed_examples

def process_test(data_dir):
    file = data_dir / 'test.json'
    test_type_to_key = {
        'clean': 'question',
        'noisy_1': 'noisy1_question',
        'noisy_2': 'noisy2_question',
        'noisy_3': 'noisy3_question',
    }

    for test_type, q_key in test_type_to_key.items():
        parsed_examples = process_file(file, q_key)
        dump_jsonl(parsed_examples, data_dir / f'test_{test_type}.json')


def process_train_and_dev(data_dir):
    train_examples = process_file(data_dir / 'train.json', 'question')
    dump_jsonl(train_examples, data_dir / f'train.json')
    dev_examples = process_file(data_dir / 'dev.json', 'question')
    dump_jsonl(dev_examples, data_dir / f'dev.json')


def preprocess(data_dir):
    print('Processing train and dev...')
    process_train_and_dev(data_dir)
    print('Processing test data...')
    process_test(data_dir)
    

if __name__ == '__main__':
    data_dir = Path('asr/cspider')
    preprocess(data_dir)
