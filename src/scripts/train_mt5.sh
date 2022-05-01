#!/bin/bash
#SBATCH -p rtx2080
#SBATCH --nodes 1
#SBATCH --gpus-per-node 2
#SBATCH --job-name=mT5-afqmc

# module load cuda/10.2

task="afqmc_balanced"
output_dir="results/$task/mt5"
data_dir="../data/AutoASR/$task"

# Command
# cmd="python3 train_mt5.py"
# cmd="train_mt5.py"
# cmd="python3 -m torch.distributed.launch --nproc_per_node=2 train_mt5.py"
cmd="torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train_mt5.py"
cmd+=" --output_dir $output_dir"
cmd+=" --data_dir $data_dir"
cmd+=" --log_interval 1"
cmd+=" --num_epochs 10"
cmd+=" --batch_size 2"
cmd+=" --grad_acc_steps 128"
cmd+=" --lr 2e-4"
# cmd+=" --resume_from_checkpoint"

logfile="$output_dir/log.txt"
mkdir -p $output_dir

# Execute
echo $cmd
echo ''
$cmd | tee $logfile
