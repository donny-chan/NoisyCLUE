#!/bin/bash
#SBATCH -p rtx2080
#SBATCH -G 1
#SBATCH --job-name ner-mb

model_path="chinese-roberta-wwm-ext"
model_path="chinese-macbert-base"
task="cluener"
task_parent="keyboard"
# task_parent="autoasr"
lr="2e-4"

output_dir="results/${task_parent}/${task}/${model_path}_lr${lr}"
data_dir="../data/${task_parent}/${task}"

# Command
cmd="python3 train_ner.py"
cmd+=" --model_path hfl/$model_path"
cmd+=" --output_dir $output_dir"
cmd+=" --data_dir $data_dir"

cmd+=" --num_epochs 10"
cmd+=" --batch_size 16"
cmd+=" --grad_acc_steps 16"
cmd+=" --lr $lr"
cmd+=" --log_interval 10"
# cmd+=" --resume_from_checkpoint"

logfile="$output_dir/log.txt"
mkdir -p $output_dir

# Execute
echo $cmd
echo ''
$cmd | tee $logfile
