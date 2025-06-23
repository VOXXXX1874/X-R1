#!/bin/bash
#SBATCH --job-name=numath
#SBATCH --mail-user=zzchen2@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/d2/spc/zzchen2/X-R1/tmp/results.txt
#SBATCH --gres=gpu:4
#SBATCH --qos=gpu
#SBATCH -p gpu_24h
#SBATCH -c 12
#SBATCH -w projgpu13

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=3 \
src/x_r1/grpo.py \
--config recipes/X_R1_1dot5B_config_math_num.yaml \
> ./output/X_R1_1dot5B_math_num.log 2>&1