#!/bin/bash
#SBATCH --job-name=sscale2
#SBATCH --mail-user=zzchen2@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/d2/spc/zzchen2/X-R1/tmp/sscale-results2.txt
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH -p gpu_2h
#SBATCH -c 6
#SBATCH -C rtx3090

CUDA_VISIBLE_DEVICES=0 python ./src/x_r1/benchmark_MVOT_latent_pencil.py \
	--model_name='./records/Qwen2.5-1.5B-pencil-xr1' \
    --dataset_name='src/cv_extraction/MATH-500/exp' \
	--output_name='./output/result_benchmark_math500'  \
	--max_output_tokens=1024 \
	--num_generation=16 \
	--max_steps=0 \
	--num_gpus=1 \
	--layer=-1 \
	--temperature=0.7 > output/benchmark_sampling-p16-s0.log 2>&1
#
#CUDA_VISIBLE_DEVICES=0 python ./src/x_r1/benchmark_MVOT_latent_pencil.py \
#	--model_name='./records/Qwen2.5-1.5B-pencil-xr1' \
#    --dataset_name='src/cv_extraction/MATH-500/exp' \
#	--output_name='./output/result_benchmark_math500'  \
#	--max_output_tokens=2048 \
#	--num_generation=16 \
#	--max_steps=1 \
#	--num_gpus=1 \
#	--layer=-1 \
#	--temperature=0.7 > output/benchmark_sampling-p16-s1.log 2>&1

CUDA_VISIBLE_DEVICES=0 python ./src/x_r1/benchmark_MVOT_latent_pencil.py \
	--model_name='./records/Qwen2.5-1.5B-pencil-xr1' \
    --dataset_name='src/cv_extraction/MATH-500/exp' \
	--output_name='./output/result_benchmark_math500'  \
	--max_output_tokens=2048 \
	--num_generation=16 \
	--max_steps=2 \
	--num_gpus=1 \
	--layer=-1 \
	--temperature=0.7 > output/benchmark_sampling-p16-s2.log 2>&1

#CUDA_VISIBLE_DEVICES=0 python ./src/x_r1/benchmark_MVOT_latent_pencil.py \
#	--model_name='./records/Qwen2.5-1.5B-pencil-xr1' \
#    --dataset_name='src/cv_extraction/MATH-500/exp' \
#	--output_name='./output/result_benchmark_math500'  \
#	--max_output_tokens=2048 \
#	--num_generation=16 \
#	--max_steps=3 \
#	--num_gpus=1 \
#	--layer=-1 \
#	--temperature=0.7 > output/benchmark_sampling-p16-s3.log 2>&1
#
#CUDA_VISIBLE_DEVICES=0 python ./src/x_r1/benchmark_MVOT_latent_pencil.py \
#	--model_name='./records/Qwen2.5-1.5B-pencil-xr1' \
#    --dataset_name='src/cv_extraction/MATH-500/exp' \
#	--output_name='./output/result_benchmark_math500'  \
#	--max_output_tokens=2048 \
#	--num_generation=16 \
#	--max_steps=4 \
#	--num_gpus=1 \
#	--layer=-1 \
#	--temperature=0.7 > output/benchmark_sampling-p16-s4.log 2>&1
#
#CUDA_VISIBLE_DEVICES=0 python ./src/x_r1/benchmark_MVOT_latent_pencil.py \
#	--model_name='./records/Qwen2.5-1.5B-pencil-xr1' \
#    --dataset_name='src/cv_extraction/MATH-500/exp' \
#	--output_name='./output/result_benchmark_math500'  \
#	--max_output_tokens=2048 \
#	--num_generation=16 \
#	--max_steps=5 \
#	--num_gpus=1 \
#	--layer=-1 \
#	--temperature=0.7 > output/benchmark_sampling-p16-s5.log 2>&1
#
#CUDA_VISIBLE_DEVICES=0 python ./src/x_r1/benchmark_MVOT_latent_pencil.py \
#	--model_name='./records/Qwen2.5-1.5B-pencil-xr1' \
#    --dataset_name='src/cv_extraction/MATH-500/exp' \
#	--output_name='./output/result_benchmark_math500'  \
#	--max_output_tokens=2048 \
#	--num_generation=16 \
#	--max_steps=6 \
#	--num_gpus=1 \
#	--layer=-1 \
#	--temperature=0.7 > output/benchmark_sampling-p16-s6.log 2>&1