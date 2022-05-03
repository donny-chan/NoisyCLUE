from pathlib import Path
import json


# label_texts = ['nonequivalent</s>', 'equivalent</s>']
label_to_id = {'nonequivalent</s>': 0, 'equivalent</s>': 1}
label_to_id = {'non_equivalent</s>': 0, 'equivalent</s> <pad>': 1}
task = 'afqmc_unbalanced'
# model_pattern = 'mt5-base_lr*'
model_pattern = '*'

def get_labels():
    test_file = Path(f'../data/AutoASR/{task}/test_clean.json')
    test_data = [json.loads(line) for line in test_file.open('r')]
    labels = [int(d['label']) for d in test_data]
    return labels


def print_table(rows, headers, types):
    def to_str(x, type):
        if x is None:
            return '-'
        if type is float:
            return f'{x:.4f}'
        return str(x)

    def _print_row(row, lens):
        for i, x in enumerate(row):
            s = f'| {x}'
            s += ' ' * (lens[i] - len(x) + 1)
            print(s, end='')
        print('|')

    def _print_hor_line(lens):
        for i, x in enumerate(lens):
            print('+', end='')
            print('-' * (x + 2), end='')
        print('+')

    rows = [[to_str(x, t) for x, t in zip(row, types)] for row in rows]
    # Get len of columns
    col_lens = [len(h) for h in headers]
    for row in rows:
        for i in range(len(col_lens)):
            col_lens[i] = max(col_lens[i], len(row[i]))

    def print_row(row): _print_row(row, col_lens)
    def print_hor_line(): _print_hor_line(col_lens)

    # Print
    print_hor_line()
    print_row(headers)
    print_hor_line()
    for row in rows:
        print_row(row)
    print_hor_line()


def get_acc(preds, labels) -> float:
    assert len(preds) == len(labels)
    correct = 0
    for a, b in zip(preds, labels):
        if a == b:
            correct += 1
    return correct / len(preds)


def get_preds(dir, is_seq2seq) -> list:
    try:
        if is_seq2seq:
            result_file = dir / 'preds_text.json'
            preds = json.load(result_file.open('r'))
            pred_ids = [label_to_id.get(x, -1) for x in preds]
            return pred_ids
        else:
            preds = dir / 'preds.txt'
            return json.load(preds.open('r'))
    except Exception:
        return None


def get_result(result_dir, labels, is_seq2seq):
    data_names = ['clean', 'noisy_1', 'noisy_2', 'noisy_3']
    result = {}
    for data_name in data_names:
        try:
            result_file = result_dir / f'test_{data_name}' / 'result.json'
            acc = json.load(result_file.open('r'))['acc']
            result[f'acc_{data_name}'] = acc
        except Exception:
            continue

    noisy_accs = [result.get(x, None) for x in ['acc_noisy_1', 'acc_noisy_2', 'acc_noisy_3']]
    if None in noisy_accs:
        return result
    result['avg'] = sum(noisy_accs) / len(noisy_accs)

    def get_worst_group_acc():
        noisy_preds = []
        noisy_dirs = sorted(result_dir.glob('test_noisy_*'))
        if len(noisy_dirs) != 3:
            return None
        for dir in noisy_dirs:
            preds = get_preds(dir, is_seq2seq)
            if preds is None:
                return None
            noisy_preds.append(preds)
        count = len(noisy_preds[0])
        correct = 0
        for i in range(count):
            if all(labels[i] == preds[i] for preds in noisy_preds):
                correct += 1
        return correct / count

    result['worst_group'] = get_worst_group_acc()
    return result


data_names = ['clean', 'noisy_1', 'noisy_2', 'noisy_3']
test_names = [f'test_{x}' for x in data_names]
labels = get_labels()

results_dir = Path('results_0') / task

# types = [str, str, float, float]
# headers = ['model', 'data', 'loss', 'acc']
headers = ['model'] + [f'acc_{x}' for x in data_names] + ['avg', 'worst_group']
types = [str] + [float] * 6
rows = []

for result_dir in sorted(results_dir.glob(model_pattern)):
    is_seq2seq = any(x in result_dir.name for x in ['mt5', 'byt5'])
    result = get_result(result_dir, labels, is_seq2seq)
    row = [result.get(h, None) for h in headers[1:]]
    row = [result_dir.name] + row
    rows.append(row)

print_table(rows, headers, types)
