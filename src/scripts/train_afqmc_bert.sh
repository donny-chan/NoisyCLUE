#!/bin/bash
#SBATCH -p rtx2080
#SBATCH -G 1
#SBATCH --job-name rb-unbalanced

task="afqmc_unbalanced"
# task_parent="asr"
# task_parent="autoasr"

# Model
model_path="chinese-roberta-wwm-ext"
# model_path="rbt4"
lr="5e-5"
model_path="chinese-macbert-base"
lr="2e-5"

# output_dir="results/${task_parent}/${task}/${model_path}_lr${lr}"
data_dir="../data/realtypo/${task}"
output_dir="results/huggingface/${task}/${model_path}_lr${lr}"
# data_dir="../data/synthetic_noise/${task}"
# output_dir="results/synthetic_noise/${task}/${model_path}_lr${lr}"

ckpt_dir="${output_dir}/checkpoint-1050"
mode="train_test"
# mode="test"

# Command
cmd="python3 train_afqmc_bert.py"
cmd+=" --model_path hfl/$model_path"
cmd+=" --output_dir $output_dir"
cmd+=" --data_dir $data_dir"

cmd+=" --num_epochs 6"
cmd+=" --batch_size 8"
cmd+=" --grad_acc_steps 32"
cmd+=" --lr $lr"
cmd+=" --log_interval 80"
cmd+=" --mode $mode"
# cmd+=" --num_examples 128"
cmd+=" --resume_from_checkpoint ${ckpt_dir}"

logfile="$output_dir/log.txt"
mkdir -p $output_dir

# Execute
echo $cmd
echo ''
$cmd | tee $logfile
