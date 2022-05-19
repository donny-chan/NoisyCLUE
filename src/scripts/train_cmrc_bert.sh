#!/bin/bash
#SBATCH -G 1
#SBATCH --job-name cmrc-rb
#__SBATCH -p rtx2080

task="cmrc2018"
# task_parent="keyboard"
# task_parent="autoasr"

model_path="chinese-roberta-wwm-ext"
model_path="chinese-macbert-base"

# Hyperparams
lr="1e-4"

output_dir="results/${task}/${model_path}_lr${lr}"
data_dir="../data/realtypo/${task}"

# Command
cmd="python3 train_cmrc_bert.py"
cmd+=" --model_path hfl/$model_path"
cmd+=" --output_dir $output_dir"
cmd+=" --data_dir $data_dir"

cmd+=" --num_epochs 10"
cmd+=" --batch_size 4"
cmd+=" --grad_acc_steps 64"
cmd+=" --lr $lr"
cmd+=" --log_interval 100"
cmd+=" --resume_from_checkpoint placeholder"

logfile="$output_dir/log.txt"
mkdir -p $output_dir

# Execute
echo $cmd
echo ''
$cmd | tee $logfile
