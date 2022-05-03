import json
from argparse import Namespace
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader
import numpy as np
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from data.afqmc import AfqmcDataset, AfqmcSeq2SeqDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dump_str(data, file: Path):
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open('w') as f:
        f.write(str(data))


def dump_args(args, file: Path):
    s = json.dumps(vars(args), indent=2, ensure_ascii=False)
    file.open('w').write(s)
    print(s)

def dump_json(data, file: Path, **kwargs):
    json.dump(data, file.open('w'), ensure_ascii=False, **kwargs)


def get_acc(preds: np.array, labels: np.array) -> float:
    return np.mean(np.argmax(preds, axis=1) == labels)


def get_param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _get_dataset(file: Path, phase: str, **kwargs) -> AfqmcDataset:
    return AfqmcDataset(file, phase, max_seq_len=512, **kwargs)


def  get_dataset(data_dir: Path, phase: str, **kwargs) -> AfqmcDataset:
    return _get_dataset(data_dir / f'{phase}.json', phase, **kwargs)
