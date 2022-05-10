from pathlib import Path

from print_utils import print_table, dump_table
from utils import load_json


task_type = 'keyboard'
task = 'cmrc2018'


def get_result(result_dir: Path, labels: dict) -> dict:
    all_result = {}
    for test_type in ['clean', 'noisy_1', 'noisy_2', 'noisy_3']:
        test_dir = result_dir / f'test_{test_type}'
        result_file = test_dir / 'result.json'
        if result_file.exists():
            result = load_json(result_file)
            all_result[f'acc_{test_type}'] = result['acc']

    def get_avg_acc():
        noisy_accs = [all_result.get(k, None) for k in [f'acc_noisy_{i}' for i in range(1, 4)]]
        if None in noisy_accs:
            return None
        return sum(noisy_accs) / len(noisy_accs)
    
    all_result['avg_acc'] = get_avg_acc()

    return all_result


headers = {
    'model': str,
    'acc_clean': float,
    'acc_noisy_1': float,
    'acc_noisy_2': float,
    'acc_noisy_3': float,
    'avg_acc': float,
}
types = list(headers.values())
headers = list(headers.keys())

results_dir = Path('results') / task_type / task
print(f'Getting results from {results_dir}')
rows = []
for subdir in sorted(results_dir.glob('*')):
    if not subdir.is_dir(): continue
    result = get_result(subdir, None)
    row = [result.get(h, None) for h in headers[1:]]
    row = [subdir.name] + row
    rows.append(row)

headers = [h.replace('_', ' ') for h in headers]
print_table(rows, headers, types)
dump_table(rows, headers, types, results_dir / 'table.tsv')
