#!/bin/bash
#SBATCH -p rtx2080
#SBATCH -G 1
#SBATCH --mem 32G
#SBATCH --job-name un-mt5

model_path="mt5-small"
lr="1e-4"

output_dir="results/un_parallel/${model_path}_lr${lr}"
data_dir="..."

# Command
cmd="python3 train_un_parallel.py"
cmd+=" --model_path google/$model_path"
cmd+=" --output_dir $output_dir"
cmd+=" --data_dir $data_dir"

cmd+=" --num_epochs 10"
cmd+=" --batch_size 4"
cmd+=" --grad_acc_steps 64"
cmd+=" --lr $lr"
cmd+=" --log_interval 40"
# cmd+=" --resume_from_checkpoint"

logfile="$output_dir/log.txt"
mkdir -p $output_dir

# Execute
echo $cmd
echo ''
$cmd | tee $logfile
