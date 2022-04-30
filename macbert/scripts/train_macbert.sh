#!/bin/bash
#SBATCH -p rtx2080
#SBATCH -G 1

task="afqmc_balanced"
output_dir="results/$task/macbert"
data_dir="../data/AutoASR/$task"

# Command
cmd="python3 train_macbert.py"
cmd+=" --output_dir $output_dir"
cmd+=" --data_dir $data_dir"
cmd+=" --log_interval 4"

logfile="$output_dir/log.txt"
mkdir -p $output_dir

# Execute
echo $cmd
$cmd | tee $logfile
