#!/bin/bash
#SBATCH -p rtx2080
#SBATCH -G 1

task="afqmc_balanced"
output_dir="results/$task/mt5"
data_dir="../data/AutoASR/$task"

# Command
cmd="python3 train_mt5.py"
cmd+=" --output_dir $output_dir"
cmd+=" --data_dir $data_dir"
cmd+=" --log_interval 1"
cmd+=" --num_examples 128"
cmd+=" --num_epochs 2"
cmd+=" --batch_size 4"
cmd+=" --grad_acc_steps 1"
cmd+=" --lr 2e-4"

logfile="$output_dir/log.txt"
mkdir -p $output_dir

# Execute
echo $cmd
$cmd | tee $logfile
