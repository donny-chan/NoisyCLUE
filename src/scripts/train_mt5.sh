#!/bin/bash
#SBATCH -p rtx2080
#SBATCH -G 1
#SBATCH --job-name=mT5-afqmc

task="afqmc_balanced"
output_dir="results/$task/mt5"
data_dir="../data/AutoASR/$task"

# Command
cmd="python3 train_mt5.py"
cmd+=" --output_dir $output_dir"
cmd+=" --data_dir $data_dir"
cmd+=" --log_interval 1"
cmd+=" --num_epochs 4"
cmd+=" --batch_size 4"
cmd+=" --grad_acc_steps 64"
cmd+=" --lr 1e-3"

logfile="$output_dir/log.txt"
mkdir -p $output_dir

# Execute
echo $cmd
echo ''
$cmd | tee $logfile
