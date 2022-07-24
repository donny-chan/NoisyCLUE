#!/bin/bash
#SBATCH -G 1
#SBATCH --mem 32G
#SBATCH --job-name 1e-3
#_SBATCH -p rtx2080

model_path="mbart-large-cc25"
lr="1e-5"

# output_dir="results/cspider_1024/${model_path}_lr${lr}"
output_dir="results/cspider_norm/${model_path}_lr${lr}"
data_dir="../data/realtypo/cspider"

# Command
cmd="python3 train_cspider.py"
cmd+=" --model_path facebook/$model_path"
cmd+=" --output_dir $output_dir"
cmd+=" --data_dir $data_dir"

cmd+=" --num_epochs 16"
cmd+=" --batch_size 1"
cmd+=" --grad_acc_steps 64"
cmd+=" --lr $lr"
cmd+=" --log_interval 256"
# cmd+=" --resume_from_checkpoint"
# cmd+=" --num_examples 1024"
# cmd+=" --mode test"

logfile="$output_dir/log.txt"
mkdir -p $output_dir

# Execute
echo $cmd
echo ''
$cmd | tee $logfile
