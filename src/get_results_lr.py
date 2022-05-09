from pathlib import Path
import json
from collections import defaultdict

from metrics import get_bin_metrics

from print_utils import print_table, dump_table


task_parent = 'keyboard'
# task_parent = 'autoasr'
task = 'afqmc_balanced'
# task = 'afqmc_unbalanced'

# Changed labels from "non_equivalent" to "nonequivalent"
results_dir = 'results_1'
# label_to_id = {
#     'nonequivalent</s>': 0, 
#     "eonequivalent</s>": 0,
#     'equivalent</s>': 1, 
#     'nquivalent</s><pad><pad><pad>': 1,
#     "equivalent</s><pad><pad><pad>" : 1,
# }

# Chinese labels: ["不等价", "等价"]
results_dir = 'results'
# results_dir = 'results_zh_verbalizer'
# label_to_id = {
#     "不等价</s>": 0,
#     "等等价</s>": 0,
#     "等价</s> <pad>": 1,
# }

# results_dir = 'results_0'
# label_to_id = {'non_equivalent</s>': 0, 'equivalent</s> <pad>': 1}

model_pattern = '*'
# model_pattern = 'mt5-base_lr*'


def get_labels():
    test_file = Path(f'../data/{task_parent}/{task}/test_clean.json')
    test_data = [json.loads(line) for line in test_file.open('r')]
    labels = [int(d['label']) for d in test_data]
    return labels


def get_acc(preds, labels) -> float:
    assert len(preds) == len(labels)
    correct = 0
    for a, b in zip(preds, labels):
        if a == b:
            correct += 1
    return correct / len(preds)


def get_preds(dir) -> list:
    preds_file = dir / 'preds.txt'
    if preds_file.exists():
        return json.load(preds_file.open())
    try:
        preds_text_file = dir / 'preds_text.json'
        if preds_text_file.exists():
            preds = json.load(preds_text_file.open('r'))
            pred_ids = [label_to_id.get(x, -1) for x in preds]
            return pred_ids
        else:
            preds = dir / 'preds.txt'
            return json.load(preds.open('r'))
    except Exception:
        return None


def get_result(result_dir, labels):
    data_names = ['clean', 'noisy_1', 'noisy_2', 'noisy_3']
    result = defaultdict()
    # for data_name in data_names:
    #     try:
    #         result_file = result_dir / f'test_{data_name}' / 'result.json'
    #         acc = json.load(result_file.open('r'))['acc']
    #         result[f'acc_{data_name}'] = acc
    #     except Exception:
    #         continue

    # noisy_accs = [result.get(x, None) for x in ['acc_noisy_1', 'acc_noisy_2', 'acc_noisy_3']]
    # if None in noisy_accs:
    #     return result

    def get_worst_group_acc():
        noisy_preds = []
        noisy_dirs = sorted(result_dir.glob('test_noisy_*'))
        if len(noisy_dirs) != 3:
            return None
        for dir in noisy_dirs:
            preds = get_preds(dir)
            if preds is None:
                return None
            noisy_preds.append(preds)
        count = len(noisy_preds[0])
        correct = 0
        for i in range(count):
            if all(labels[i] == preds[i] for preds in noisy_preds):
                correct += 1
        return correct / count


    def get_avg_acc(results):
        noisy_accs = [result.get(x, None) for x in ['acc_noisy_1', 'acc_noisy_2', 'acc_noisy_3']]
        if None in noisy_accs:
            return None
        return sum(noisy_accs) / len(noisy_accs)
    

    def get_metrics(result: dict, dir):
        for data_name in data_names:
            preds = get_preds(dir / f'test_{data_name}')
            if preds == None:
                continue
            metrics = get_bin_metrics(labels, preds)
            if data_name == 'clean':
                result['macro_f1_{}'.format(data_name)] = metrics.get('macro_f1', None)
            result[f'f1_0_{data_name}'] = metrics.get('f1_0', None)
            result[f'f1_1_{data_name}'] = metrics.get('f1_1', None)
            result[f'acc_{data_name}'] = metrics.get('acc', None)

    get_metrics(result, result_dir)
    result['worst_group'] = get_worst_group_acc()
    result['avg'] = get_avg_acc(result)
    return result


data_names = ['clean', 'noisy_1', 'noisy_2', 'noisy_3']
test_names = [f'test_{x}' for x in data_names]
labels = get_labels()
results_dir = Path(results_dir) / task_parent / task

headers = {
    'model': str,
    'acc_clean': float,
    'acc_noisy_1': float,
    'acc_noisy_2': float,
    'acc_noisy_3': float,
    'avg': float,
    'worst_group': float,
    'macro_f1_clean': float,
    'f1_0_clean': float,
    'f1_1_clean': float,
    'f1_0_noisy_1': float,
    'f1_1_noisy_1': float,
    'f1_0_noisy_2': float,
    'f1_1_noisy_2': float,
    'f1_0_noisy_3': float,
    'f1_1_noisy_3': float,
}
types = list(headers.values())
headers = list(headers.keys())

print(f'Getting results from {results_dir}')
rows = []
subdirs = sorted(d for d in results_dir.glob(model_pattern) if d.is_dir())
for result_dir in subdirs:
    result = get_result(result_dir, labels)
    row = [result.get(h, None) for h in headers[1:]]
    row = [result_dir.name] + row
    rows.append(row)

headers = [h.replace('_', ' ') for h in headers]
print_table(rows, headers, types)
dump_table(rows, headers, types, results_dir / 'table.tsv')
