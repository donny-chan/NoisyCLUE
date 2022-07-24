from pathlib import Path
import json
from collections import defaultdict

from afqmc.metrics import get_bin_metrics

from print_utils import print_table, dump_table


def get_labels(task):
    test_file = Path(f'../data/realtypo/{task}/test_clean.json')
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


def get_result(result_dir: Path, test_name: str, labels: list) -> dict:
    result = defaultdict()
    
    total_pos = labels.count(1)
    total_neg = labels.count(0)
    print('total:', total_pos, total_neg)

    def get_pos_neg_acc(labels, preds):
        correct_pos = 0
        correct_neg = 0
        for label, pred in zip(labels, preds):
            if label == pred:
                if label == 0:
                    correct_neg += 1
                elif label == 1:
                    correct_pos += 1
                else:
                    raise ValueError
        acc_pos = correct_pos / total_pos
        acc_neg = correct_neg / total_neg
        # print('# correct:', correct_pos, correct_neg)
        return acc_pos, acc_neg

    def get_worst_group_acc(result):
        noisy_preds = []
        noisy_dirs = sorted(result_dir.glob(f'test_noisy_{noise_type}_*'))
        if len(noisy_dirs) != 3:
            return None
        for dir in noisy_dirs:
            preds = get_preds(dir)
            if preds is None:
                return None
            noisy_preds.append(preds)
        count = len(noisy_preds[0])
        same_preds = [None] * count
        for i in range(count):
            if all(preds[i] == noisy_preds[0][i] for preds in noisy_preds):
                same_preds[i] = noisy_preds[0][i]
        acc_pos, acc_neg = get_pos_neg_acc(labels, same_preds)
        result['pos_worst'] = acc_pos
        result['neg_worst'] = acc_neg

    def get_avg_acc(result):
        for pos_neg in ['pos', 'neg']:
            noisy_accs = [result.get(pos_neg + '_' + x, None) for x in ['acc_noisy_1', 'acc_noisy_2', 'acc_noisy_3']]
            if None in noisy_accs:
                return None
            result[f'{pos_neg}_avg'] = sum(noisy_accs) / len(noisy_accs)

    def get_metrics(result: dict, dir):
        # # Clean
        # preds = get_preds(dir / 'test_clean')
        # if preds:
        #     acc_pos, acc_neg = get_pos_neg_acc(labels, preds)
        #     result['pos_acc_clean'] = acc_pos
        #     result['neg_acc_clean'] = acc_neg
        
        # Noisy
        # for i in range(1, 4):
        # test_name = f'test_noisy_{noise_type}_{i}'
        # test_name = f'test_synthetic_noise_{noise_type}'
        preds = get_preds(dir / test_name)
        if preds == None:
            return {}
        acc_pos, acc_neg = get_pos_neg_acc(labels, preds)
        result['pos_acc'] = acc_pos
        result['neg_acc'] = acc_neg

    get_metrics(result, result_dir)
    # get_worst_group_acc(result)
    # get_avg_acc(result)
    return result


def get_table(task, results_dir, model_pattern='*'):
    labels = get_labels(task)
    results_dir = Path(results_dir) / task

    headers = {
        'model': str,
        'test_name': str,
        'pos_acc': float,
        'neg_acc': float,
        # 'pos_acc_clean': float,
        # 'pos_acc_noisy_1': float,
        # 'pos_acc_noisy_2': float,
        # 'pos_acc_noisy_3': float,
        # 'pos_glyph_50': float,
        # 'pos_glyph_100': float,
        # 'pos_phonetic_10': float,
        # 'pos_avg': float,
        # 'pos_worst': float,
        # 'neg_acc_clean': float,
        # 'neg_acc_noisy_1': float,
        # 'neg_acc_noisy_2': float,
        # 'neg_acc_noisy_3': float,
        # 'neg_avg': float,
        # 'neg_worst': float,
    }
    types = list(headers.values())
    headers = list(headers.keys())

    print(f'Getting results from {results_dir}')
    rows = []
    model_dirs = sorted(d for d in results_dir.glob(model_pattern) if d.is_dir())
    for model_dir in model_dirs:
        for test_name in [
            'test_clean',
            'test_synthetic_noise_glyph_50',
            'test_synthetic_noise_glyph_100',
            'test_synthetic_noise_phonetic_0',
            'test_synthetic_noise_phonetic_10',
            'test_synthetic_noise_phonetic_20',
            'test_synthetic_noise_phonetic_30',
            'test_synthetic_noise_phonetic_40',
            'test_synthetic_noise_phonetic_50',
        ]:
            result = get_result(model_dir, test_name, labels)
            # print(result)
            # exit()
            row = [result.get(h, None) for h in headers[2:]]
            row = [model_dir.name, test_name] + row
            rows.append(row)

    # headers = [h.replace('_', ' ') for h in headers]
    print_table(rows, headers, types)
    dump_table(rows, headers, types, results_dir / f'table_synthetic_noise.tsv')


def main():
    # task = 'afqmc_balanced'
    task = 'afqmc_unbalanced'
    results_dir = 'results/huggingface'
    model_pattern = '*'  # get all models
    # for noise_type in ['keyboard', 'asr']:
        # print(f'Getting results for {noise_type}')
    get_table(task, results_dir, model_pattern)
    

if __name__ == '__main__':
    main()