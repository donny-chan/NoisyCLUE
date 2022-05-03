#!/bin/bash
#SBATCH -p rtx2080
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --job-name=mT5-balanced

lr="3e-4"
task="afqmc_balanced"
model_name="mt5-base"

data_dir="../data/AutoASR/$task"
output_dir="results/$task/${model_name}_lr${lr}"
num_gpus=1

# Command
cmd="python3 train_mt5.py"
# cmd="torchrun --nnodes=1 --nproc_per_node=$num_gpus --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train_mt5.py"
cmd+=" --model_path google/$model_name"
cmd+=" --output_dir $output_dir"
cmd+=" --data_dir $data_dir"
cmd+=" --log_interval 20"
cmd+=" --num_epochs 10"
cmd+=" --batch_size 16"
cmd+=" --grad_acc_steps 16"
cmd+=" --lr ${lr}"
# cmd+=" --resume_from_checkpoint"

logfile="$output_dir/log.txt"
mkdir -p $output_dir

# Execute
echo $cmd
echo ''
$cmd | tee $logfile
