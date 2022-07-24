from pathlib import Path
import json


def load_labels(data_dir: Path) -> list:
    key = 'query'
    file_clean = data_dir / 'test_clean.json'
    queries = [json.loads(line)['query'].lower() for line in open(file_clean, 'r')]
    return queries


def load_texts(file) -> list:
    lines = []
    for line in open(file, 'r', encoding='utf8'):
        lines.append(line.strip()[1:-1].lower())
    return lines


def normalize(text):
    # text = text.lower()
    # for c in ',()<>!@#$%^&*':
        # text = text.replace(c, ' ' + c + ' ')
    text = text.strip().split()
    return ' '.join(text)


def get_score(output_dir: Path, labels: list):
    # print(f'Getting score for {output_dir}')
    file_preds = output_dir / 'preds_text.json'
    preds = load_texts(file_preds)
    # print(preds[:5])
    # print(labels[:5])
    # exit()
    
    # Compute metrics
    correct = 0
    for i, (label, pred) in enumerate(zip(labels, preds)):
        label = normalize(label)
        pred = normalize(pred)
        if label == pred:
            # print(i, label)
            correct += 1
    acc = correct / len(labels)
    
    # Get loss
    loss = json.load(open(output_dir / 'result.json', 'r'))['loss']
    
    print(f'  {output_dir.name:<22}\t{100*acc:.2f}\t{loss:.4f}')
    
    
def get_all_scores(output_dir, labels):
    # Clean
    get_score(output_dir / 'test_clean', labels)
    
    # Noisy
    for noise_type in ['keyboard', 'asr']:
        for i in range(1, 4):
            get_score(output_dir / f'test_noisy_{noise_type}_{i}', labels)


if __name__ == '__main__':
    cspider_dir = Path('../../data/realtypo/cspider')
    labels = load_labels(cspider_dir)
    
    # lr = "2e-4"
    # model_path = f'mbart-large-cc25_lr{lr}'
    result_dir = Path('../results/cspider_norm')
    # output_dir = result_dir / '{model_path}'
    
    for output_dir in result_dir.glob('*'):
        if not output_dir.is_dir(): continue
        print(output_dir)
        get_all_scores(output_dir, labels)
