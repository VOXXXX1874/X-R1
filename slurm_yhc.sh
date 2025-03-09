#!/bin/bash
#SBATCH --job-name=xr1
#SBATCH --mail-user=zzchen2@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/jcheng3/hcyang/X-R1/tmp/results.txt
#SBATCH --gres=gpu:4
#SBATCH -p jcheng_gpu_72h
#SBATCH -c 24

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/zero3.yaml --num_processes=3 src/x_r1/grpo.py --config recipes/X_R1_3B_supervised_1dot5B_config.yaml > ./output/X_R1_3B_supervised_1dot5B.log 2>&1