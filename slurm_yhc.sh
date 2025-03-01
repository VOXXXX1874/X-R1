#!/bin/bash
#SBATCH --job-name=xr1
#SBATCH --mail-user=zzchen2@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/jcheng3/hcyang/X-R1/tmp/results.txt
#SBATCH --gres=gpu:4
#SBATCH -p jcheng_gpu_72h
#SBATCH -c 16

export LD_LIBRARY_PATH=/research/d2/spc/zzchen2/anaconda/envs/xr1/lib:$LD_LIBRARY_PATH

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/zero3.yaml --num_processes=3 src/x_r1/grpo.py --config recipes/X_R1_zero_3B_config.yaml > ./output/x_r1_3B_sampling.log 2>&1