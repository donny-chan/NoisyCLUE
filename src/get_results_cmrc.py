from pathlib import Path

from print_utils import print_table, dump_table
from utils import load_json, load_jsonl


task_type = 'keyboard'
# task = 'cmrc2018_7762'
# task = 'cmrc2018_6653'
task = 'cmrc2018'


def get_id2labels() -> dict:
    examples_file = Path(f'../data/realtypo/{task}/test_clean.json')
    examples = load_jsonl(examples_file)
    id2labels = {}
    for example in examples:
        labels = [ans['text'] for ans in example['answers']]
        id2labels[example['id']] = labels
    return id2labels


def get_result(result_dir: Path, noise_type: str, id2labels: dict) -> dict:
    def get_worst_group_acc(result):
        all_preds = []  # (3, N)
        for i in range(1, 4):
            test_dir = result_dir / f'test_noisy_{noise_type}_{i}'
            preds_file = test_dir / 'preds.json'
            if preds_file.exists():
                all_preds.append(load_json(preds_file))
        if len(all_preds) != 3: return None
        
        # Collapse three preds into one
        ids = id2labels.keys()
        correct = 0
        for eid in ids:
            if all(all_preds[i][eid] in id2labels[eid] for i in range(3)):
                correct += 1    
        result['worst'] = correct / len(id2labels)

    def get_avg_acc(result):
        noisy_accs = [result.get(k, None) for k in [f'acc_noisy_{i}' for i in range(1, 4)]]
        if None in noisy_accs:
            return None
        result['avg'] = sum(noisy_accs) / len(noisy_accs)
   
    def get_noisy_accs(result):
        for i in range(1, 4):
            test_dir = result_dir / f'test_noisy_{noise_type}_{i}'
            result_file = test_dir / 'result.json'
            if result_file.exists():
                result[f'acc_noisy_{i}'] = load_json(result_file)['acc']
    
    # Clean
    result = {}
    result_file = result_dir / 'test_clean' / 'result.json'
    if result_file.exists():
        result[f'acc_clean'] = load_json(result_file)['acc']
    get_noisy_accs(result)
    get_avg_acc(result)
    get_worst_group_acc(result)

    return result


def get_table(results_dir, noise_type, id2labels):
    headers = {
        'model': str,
        'acc_clean': float,
        'acc_noisy_1': float,
        'acc_noisy_2': float,
        'acc_noisy_3': float,
        'avg': float,
        'worst': float,
    }
    types = list(headers.values())
    headers = list(headers.keys())

    rows = []
    for subdir in sorted(results_dir.glob('*')):
        if not subdir.is_dir(): continue
        result = get_result(subdir, noise_type, id2labels)
        row = [result.get(h, None) for h in headers[1:]]
        row = [subdir.name] + row
        rows.append(row)

    headers = [h.replace('_', ' ') for h in headers]
    print_table(rows, headers, types)
    dump_table(rows, headers, types, results_dir / f'table_{noise_type}.tsv')


if __name__ == '__main__':
    results_dir = Path('results') / task
    print(f'Getting results from {results_dir}')
    id2labels = get_id2labels()
    
    for noise_type in ['keyboard', 'asr']:
        print(f"*** Getting results for {noise_type} ***")
        get_table(results_dir, noise_type, id2labels)
    
