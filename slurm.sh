#!/bin/bash
#SBATCH --job-name=pencil
#SBATCH --mail-user=zzchen2@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/d2/spc/zzchen2/X-R1/tmp/results.txt
#SBATCH --gres=gpu:2
#SBATCH --qos=gpu
#SBATCH -p gpu_24h
#SBATCH -c 12
#SBATCH -w projgpu13

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=2 src/x_r1/sft.py \
--config recipes/Pencil_1dot5B_config.yaml \
> ./output/pencil_1dot5B_sampling.log 2>&1