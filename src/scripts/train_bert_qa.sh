#!/bin/bash
#SBATCH -p rtx2080
#SBATCH -G 1
#SBATCH --job-name cmrc-rb

model_path="chinese-roberta-wwm-ext"
model_path="chinese-macbert-base"
task="cmrc2018"
task_parent="keyboard"
# task_parent="autoasr"
lr="2e-5"

output_dir="results/${task_parent}/${task}/${model_path}_lr${lr}"
data_dir="../data/${task_parent}/${task}"

# Command
cmd="python3 train_bert_qa.py"
cmd+=" --model_path hfl/$model_path"
cmd+=" --output_dir $output_dir"
cmd+=" --data_dir $data_dir"

cmd+=" --num_epochs 10"
cmd+=" --batch_size 8"
cmd+=" --grad_acc_steps 32"
cmd+=" --lr $lr"
cmd+=" --log_interval 10"
# cmd+=" --resume_from_checkpoint"

logfile="$output_dir/log.txt"
mkdir -p $output_dir

# Execute
echo $cmd
echo ''
$cmd | tee $logfile
