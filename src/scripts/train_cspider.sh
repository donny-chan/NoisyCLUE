#!/bin/bash
#SBATCH -p rtx2080
#SBATCH -G 1
#SBATCH --mem 32G
#SBATCH --job-name un-mt5b

model_path="mt5-base"
lr="3e-5"

output_dir="results/cspider/${model_path}_const-lr${lr}"
data_dir="..."

# Command
cmd="python3 train_cspider.py"
cmd+=" --model_path google/$model_path"
cmd+=" --output_dir $output_dir"
cmd+=" --data_dir $data_dir"

cmd+=" --num_epochs 1"
cmd+=" --batch_size 1"
cmd+=" --grad_acc_steps 32"
cmd+=" --lr $lr"
cmd+=" --log_interval 20"
# cmd+=" --resume_from_checkpoint"

logfile="$output_dir/log.txt"
mkdir -p $output_dir

# Execute
echo $cmd
echo ''
$cmd | tee $logfile
