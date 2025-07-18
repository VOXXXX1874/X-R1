#!/bin/bash
#SBATCH --job-name=augment
#SBATCH --mail-user=zzchen2@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/d2/spc/zzchen2/X-R1/tmp/augment_results.txt
#SBATCH --gres=gpu:1
#SBATCH --reser=jcheng_gpu_301
#SBATCH -c 4
#SBATCH -p gpu_24h

python /research/d2/spc/zzchen2/X-R1/src/cv_extraction/XR1-750/extend.py