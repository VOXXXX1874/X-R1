#!/bin/bash
#SBATCH --job-name=compare
#SBATCH --mail-user=zzchen2@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/d2/spc/zzchen2/X-R1/tmp/compare.txt
#SBATCH --gres=gpu:7
#SBATCH --qos=gpu
#SBATCH -p gpu_24h
#SBATCH -c 24
#SBATCH -w projgpu13

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=6 \
src/x_r1/grpo.py \
--config recipes/X_R1_1dot5B_config_math_num-1.yaml \
> ./output/X_R1_1dot5B_math_num-1.log 2>&1