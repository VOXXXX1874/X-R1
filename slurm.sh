#!/bin/bash
#SBATCH --job-name=regexMath
#SBATCH --mail-user=zzchen2@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/d2/spc/zzchen2/X-R1/tmp/results.txt
#SBATCH --gres=gpu:4
#SBATCH --qos=gpu
#SBATCH -p gpu_24h
#SBATCH -c 24
#SBATCH -w projgpu14

ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=3 src/x_r1/grpo.py \
--config recipes/X_R1_1dot5B_config_math_regex.yaml \
> ./output/x_r1_1dot5B_sampling.log 2>&1