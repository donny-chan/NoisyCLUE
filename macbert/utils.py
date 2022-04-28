from pathlib import Path

def dump_str(data, file: Path):
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open('w') as f:
        f.write(str(data))