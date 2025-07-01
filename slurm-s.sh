#!/bin/bash
#SBATCH --job-name=compare
#SBATCH --mail-user=zzchen2@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/d2/spc/zzchen2/X-R1/tmp/compare.txt
#SBATCH --gres=gpu:5
#SBATCH --reser=jcheng_gpu_301
#SBATCH -c 24
#SBATCH -p gpu_24h

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=4 \
src/x_r1/grpo.py \
--config recipes/X_R1_1dot5B_config_math_num-1.yaml \
> ./output/X_R1_1dot5B_math_num-1.log 2>&1