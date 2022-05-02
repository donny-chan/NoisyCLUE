#!/bin/bash
#SBATCH -p rtx2080
#SBATCH -G 1
#SBATCH --job-name mb-afqmc

model_path="chinese-roberta-wwm-ext"
task="afqmc_unbalanced"
output_dir="results/$task/$model_path"
data_dir="../data/AutoASR/$task"
model_path="hfl/$model_path"

# Command
cmd="python3 train_macbert.py"
cmd+=" --model_path $model_path"
cmd+=" --output_dir $output_dir"
cmd+=" --data_dir $data_dir"
cmd+=" --log_interval 5"
cmd+=" --num_epochs 6"

logfile="$output_dir/log.txt"
mkdir -p $output_dir

# Execute
echo $cmd
echo ''
$cmd | tee $logfile
