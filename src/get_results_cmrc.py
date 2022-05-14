from pathlib import Path

from print_utils import print_table, dump_table
from utils import load_json, load_jsonl


task_type = 'keyboard'
task = 'cmrc2018'


def get_id2labels() -> dict:
    examples_file = Path('../data/keyboard/cmrc2018/cmrc2018_test_clean.json')
    examples = load_jsonl(examples_file)
    id2labels = {}
    for example in examples:
        labels = [ans['text'] for ans in example['answers']]
        id2labels[example['id']] = labels
    return id2labels


def get_result(result_dir: Path, id2labels: dict) -> dict:
    all_result = {}
    phase2preds = {}

    for phase in ['clean', 'noisy_1', 'noisy_2', 'noisy_3']:
        test_dir = result_dir / f'test_{phase}'
        result_file = test_dir / 'result.json'
        if result_file.exists():
            result = load_json(result_file)
            all_result[f'acc_{phase}'] = result['acc']
        
        # Get preds for worst group acc
        preds_file = test_dir / 'preds.json'
        if preds_file.exists():
            phase2preds[phase] = load_json(preds_file)
    
    def get_worst_group_acc():
        for phase in ['clean', 'noisy_1', 'noisy_2', 'noisy_3']:
            test_dir = result_dir / f'test_{phase}'
            preds_file = test_dir / 'preds.json'
            if preds_file.exists():
                phase2preds[phase] = load_json(preds_file)
            else:
                return None
        correct = 0
        for eid, labels in id2labels.items():
            all_correct = True
            for phase, preds in phase2preds.items():
                if preds[eid] not in labels:
                    all_correct = False
                    break
            if all_correct:
                correct += 1
        return correct / len(id2labels)


    def get_avg_acc():
        noisy_accs = [all_result.get(k, None) for k in [f'acc_noisy_{i}' for i in range(1, 4)]]
        if None in noisy_accs:
            return None
        return sum(noisy_accs) / len(noisy_accs)
    
    all_result['avg_acc'] = get_avg_acc()
    all_result['worst_group_acc'] = get_worst_group_acc()

    return all_result


headers = {
    'model': str,
    'acc_clean': float,
    'acc_noisy_1': float,
    'acc_noisy_2': float,
    'acc_noisy_3': float,
    'avg_acc': float,
    'worst_group_acc': float,
}
types = list(headers.values())
headers = list(headers.keys())

results_dir = Path('results') / task_type / task
print(f'Getting results from {results_dir}')

id2labels = get_id2labels()

rows = []
for subdir in sorted(results_dir.glob('*')):
    if not subdir.is_dir(): continue
    result = get_result(subdir, id2labels)
    row = [result.get(h, None) for h in headers[1:]]
    row = [subdir.name] + row
    rows.append(row)

headers = [h.replace('_', ' ') for h in headers]
print_table(rows, headers, types)
dump_table(rows, headers, types, results_dir / 'table.tsv')
