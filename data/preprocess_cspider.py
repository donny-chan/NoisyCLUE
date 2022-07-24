from pathlib import Path
from utils import load_jsonl, dump_jsonl, load_json, dump_json


def process_file(file, question_key):
    examples = load_jsonl(file)
    parsed_examples = []
    for example in examples:
        parsed_examples.append({
            'db_id': example['db_id'],
            'question': example[question_key],
            'query': ' '.join(example['query_toks']),
        })
    return parsed_examples

def process_test(data_dir):
    print('Processing test data...')
    file = data_dir / 'cspider_test.json'
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
    print('Processing train and dev...')
    train_examples = process_file(data_dir / 'cspider_train.json', 'question')
    dump_jsonl(train_examples, data_dir / f'train.json')
    dev_examples = process_file(data_dir / 'cspider_dev.json', 'question')
    dump_jsonl(dev_examples, data_dir / f'dev.json')


def process_tables(data_dir):
    print('Preprocess tables...')
    tables_data = load_json(data_dir / 'tables.json')
    schemas = {}
    for table_data in tables_data:
        tables = table_data['table_names_original']
        columns = table_data['column_names_original']
        db_id = table_data['db_id']
        
        assert columns[0][1] == '*'
        assert columns[0][0] == -1
        assert len(tables) - 1 == columns[-1][0], f'{db_id}\n{tables}\n{columns}'
        
        schema = {}
        for table_id, col_name in columns[1:]:
            table = tables[table_id]
            if table not in schema:
                schema[table] = []
            schema[table].append(col_name)
        schemas[db_id] = schema
    file_dst = data_dir / 'schemas.json'
    print(f'Dumping {len(schemas)} schemas to {file_dst}')
    dump_json(schemas, file_dst, indent=2)
    # print(schemas[:5])
    

def preprocess(data_dir):
    process_train_and_dev(data_dir)
    process_test(data_dir)
    # process_tables(data_dir)
    

if __name__ == '__main__':
    data_dir = Path('keyboard/cspider')
    preprocess(data_dir)
