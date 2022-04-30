from pathlib import Path
import numpy as np
import random
import torch
import json


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