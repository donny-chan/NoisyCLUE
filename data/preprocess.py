'''
Steps:
1. Rename finals in `keyboard` and `asr` directory.
2. Preprocess into desired format, and split noisy examples into three different files.
3. Copy to `realtypo` directory.

'''

from pathlib import Path
import os
from shutil import copy, copytree

noise_types = ['keyboard', 'asr']
test_type = ['clean', 'noisy_1', 'noise_2', 'noisy_3']
tasks = ['afqmc_balanced', 'afqmc_unbalanced', 'cmrc2018', 'nmt', 'cspider', 'cluener']

# Rename files
print('*** Renaming files ***')
for noise_type in noise_types:
    data_dir = Path(noise_type)
    for task in tasks:
        task_dir = data_dir / task
        print('Renaming:', task_dir)
        for file in task_dir.glob('*.json'):
            filename = file.name.replace(task + '_', '')
            dst_file = task_dir / filename
            # print(f'{file} -> {dst_file}')
            os.rename(file, dst_file)


# Preprocess data
print('---------------------------')
print('*** Preprocess data ***')
import preprocess_afqmc
import preprocess_cmrc2018
import preprocess_cspider
import preprocess_nmt

for noise_type in ['keyboard', 'asr']:
    print('>> Processing noise:', noise_type)
    for task, process in [
        # 'afqmc_balanced', 
        ('afqmc_unbalanced', preprocess_afqmc.preprocess),
        ('cmrc2018', preprocess_cmrc2018.preprocess),
        ('cspider', preprocess_cspider.preprocess),
        ('nmt', preprocess_nmt.preprocess),
        ]:
        data_dir = Path(noise_type, task)
        print('>> Processing:', data_dir)
        process(data_dir)
    print('---------')


# Copy noisy test files
dst_dir = Path('realtypo')
print('---------------------------')
print(f'*** Copying files to {dst_dir} ***')
print('destination dir:', dst_dir)
for noise_type in noise_types:
    noise_dir = Path(noise_type)
    print('Copying from:', noise_dir)
    for task in tasks:
        task_dir = noise_dir / task
        dst_task_dir = dst_dir / task
        dst_task_dir.mkdir(exist_ok=True, parents=True)
        for file in task_dir.glob('test_noisy_*'):
            dst_file = dst_task_dir / file.name.replace('noisy', f'noisy_{noise_type}')
            print(f'{file} -> {dst_file}')
            copy(file, dst_file)
        # Copy other files
        print('Copying other files')
        for file in task_dir.glob('*'):
            if 'test' in file.name: continue
            dst_file = dst_task_dir / file.name
            if file.is_dir():
                copytree(file, dst_file, dirs_exist_ok=True)
            else:
                copy(file, dst_file)
print('done')
