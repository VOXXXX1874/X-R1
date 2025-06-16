#!/bin/bash
#SBATCH --job-name=pscale
#SBATCH --mail-user=zzchen2@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/d2/spc/zzchen2/X-R1/tmp/pscale-results.txt
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH -p gpu_2h
#SBATCH -c 6
#SBATCH -C rtx3090

#CUDA_VISIBLE_DEVICES=0 python ./src/x_r1/benchmark_MVOT_latent_pencil.py \
#	--model_name='./records/Qwen2.5-1.5B-pencil-xr1' \
#    --dataset_name='src/cv_extraction/MATH-500/exp' \
#	--output_name='./output/result_benchmark_math500'  \
#	--max_output_tokens=1024 \
#	--num_generation=4 \
#	--max_steps=8 \
#	--num_gpus=1 \
#	--layer=-1 \
#	--temperature=0.7 > output/benchmark_sampling-p4-s8.log 2>&1

#CUDA_VISIBLE_DEVICES=0 python ./src/x_r1/benchmark_MVOT_latent_pencil.py \
#	--model_name='./records/Qwen2.5-1.5B-pencil-xr1' \
#    --dataset_name='src/cv_extraction/MATH-500/exp' \
#	--output_name='./output/result_benchmark_math500'  \
#	--max_output_tokens=2048 \
#	--num_generation=8 \
#	--max_steps=8 \
#	--num_gpus=1 \
#	--layer=-1 \
#	--temperature=0.7 > output/benchmark_sampling-p8-s8.log 2>&1

CUDA_VISIBLE_DEVICES=0 python ./src/x_r1/benchmark_MVOT_latent_pencil.py \
	--model_name='./records/Qwen2.5-1.5B-pencil-xr1' \
    --dataset_name='src/cv_extraction/MATH-500/exp' \
	--output_name='./output/result_benchmark_math500'  \
	--max_output_tokens=1024 \
	--num_generation=16 \
	--max_steps=8 \
	--num_gpus=1 \
	--layer=-1 \
	--temperature=0.7 > output/benchmark_sampling-p16-s8.log 2>&1
#
#CUDA_VISIBLE_DEVICES=0 python ./src/x_r1/benchmark_MVOT_latent_pencil.py \
#	--model_name='./records/Qwen2.5-1.5B-pencil-xr1' \
#    --dataset_name='src/cv_extraction/MATH-500/exp' \
#	--output_name='./output/result_benchmark_math500'  \
#	--max_output_tokens=1024 \
#	--num_generation=32 \
#	--max_steps=8 \
#	--num_gpus=1 \
#	--layer=-1 \
#	--temperature=0.7 > output/benchmark_sampling-p32-s8.log 2>&1
#
#CUDA_VISIBLE_DEVICES=0 python ./src/x_r1/benchmark_MVOT_latent_pencil.py \
#	--model_name='./records/Qwen2.5-1.5B-pencil-xr1' \
#    --dataset_name='src/cv_extraction/MATH-500/exp' \
#	--output_name='./output/result_benchmark_math500'  \
#	--max_output_tokens=1024 \
#	--num_generation=64 \
#	--max_steps=8 \
#	--num_gpus=1 \
#	--layer=-1 \
#	--temperature=0.7 > output/benchmark_sampling-p64-s8.log 2>&1