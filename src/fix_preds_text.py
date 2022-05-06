import json
from pathlib import Path


folder = Path('results_1/afqmc_balanced/byt5-base_lr1e-4')
dir_names = [f'test_{x}' for x in ['clean', 'noisy_1', 'noisy_2', 'noisy_3']]

for dir_name in dir_names:
    print(dir_name)
    pred_file = folder / dir_name / 'preds_text.json'
    formatted_pred_file = folder / dir_name / 'preds_text_formatted.json'
    preds = json.load(pred_file.open('r'))
    json.dump(preds, formatted_pred_file.open('w'), ensure_ascii=False, indent=2)
    