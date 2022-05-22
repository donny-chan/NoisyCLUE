#!/bin/bash
#SBATCH -G 1
#SBATCH --mem 32G
#SBATCH --job-name csp-mt5b
#_SBATCH -p rtx2080

model_path="mbart-large-cc25"
lr="1e-4"

output_dir="results/cspider_256/${model_path}_lr${lr}"
data_dir="../data/realtypo/cspider"

# Command
cmd="python3 train_cspider.py"
cmd+=" --model_path facebook/$model_path"
cmd+=" --output_dir $output_dir"
cmd+=" --data_dir $data_dir"

cmd+=" --num_epochs 6"
cmd+=" --batch_size 4"
cmd+=" --grad_acc_steps 32"
cmd+=" --lr $lr"
cmd+=" --log_interval 1"
# cmd+=" --resume_from_checkpoint"
cmd+=" --num_examples 256"

logfile="$output_dir/log.txt"
mkdir -p $output_dir

# Execute
echo $cmd
echo ''
$cmd | tee $logfile
