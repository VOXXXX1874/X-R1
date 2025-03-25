
ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=3 src/x_r1/grpo.py \
--config recipes/X_R1_zero_0dot5B_config.yaml \
> ./output/x_r1_0dot5B_sampling.log 2>&1



ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=3 src/x_r1/grpo.py \
--config recipes/X_R1_zero_1dot5B_config.yaml \
> ./output/x_r1_1dot5B_sampling.log 2>&1



ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=3 src/x_r1/grpo.py \
--config recipes/X_R1_zero_3B_config.yaml \
> ./output/x_r1_3B_sampling.log 2>&1

# 1 3060 6GB
ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero1.yaml \
--num_processes=1 \
src/x_r1/grpo.py \
--config recipes/X_R1_0dot5B_experimental_config.yaml \
> ./output/x_r1_experimental_sampling.log 2>&1

# remember export LD_LIBRARY_PATH=/research/d2/spc/zzchen2/anaconda/envs/xr1/lib:$LD_LIBRARY_PATH
# If the ubuntu version is lower than 20.04, then the offload_param in zero3.yaml should be set to none

# > 4 3090/4090
ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=3 src/x_r1/grpo.py \
--config recipes/X_R1_1dot5B_config.yaml \
> ./output/x_r1_1dot5B_sampling.log 2>&1

ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=3 src/x_r1/grpo.py \
--config recipes/X_R1_3B_config.yaml \
> ./output/x_r1_3B_sampling.log 2>&1

ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=4 src/x_r1/grpo.py \
--config recipes/X_R1_1dot5B_config_gsc.yaml \
> ./output/x_r1_1dot5B_sampling.log 2>&1

# supervised
ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=2 src/x_r1/grpo.py \
--config recipes/X_R1_3B_supervised_0dot5B_config.yaml \
> ./output/X_R1_3B_supervised_0dot5B.log 2>&1

ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=3 src/x_r1/grpo.py \
--config recipes/X_R1_3B_supervised_1dot5B_config.yaml \
> ./output/X_R1_3B_supervised_1dot5B.log 2>&1

ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=6 src/x_r1/grpo.py \
--config recipes/X_R1_7B_supervised_1dot5B_config.yaml \
> ./output/X_R1_7B_supervised_1dot5B.log 2>&1

# benchmark

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./src/x_r1/benchmark.py \
	--model_name='records/X-R1-3B_0' \
    --dataset_name='HuggingFaceH4/MATH-500' \
	--output_name='./output/result_benchmark_math500'  \
	--max_output_tokens=2048 \
	--num_gpus=4 > output/benchmark_sampling.log 2>&1

CUDA_VISIBLE_DEVICES=0 python ./src/x_r1/benchmark.py \
	--model_name='records/X-R1-3B-7500-epoch1' \
    --dataset_name='HuggingFaceH4/MATH-500' \
	--output_name='./output/result_benchmark_math500'  \
	--max_output_tokens=2048 \
	--num_gpus=1 > output/benchmark_sampling.log 2>&1

CUDA_VISIBLE_DEVICES=0 python ./src/x_r1/benchmark.py \
	--model_name='Qwen/Qwen2.5-0.5B-Instruct' \
    --dataset_name='HuggingFaceH4/MATH-500' \
	--output_name='./output/result_benchmark_math500'  \
	--max_output_tokens=2048 \
	--num_gpus=1 \
	--reward_function='eval_answer_reward' \
	--tag False > output/benchmark_sampling.log 2>&1

CUDA_VISIBLE_DEVICES=0 python ./src/x_r1/benchmark.py \
	--model_name='Qwen/Qwen2.5-0.5B-Instruct' \
    --dataset_name='HuggingFaceH4/MATH-500' \
	--output_name='./output/result_benchmark_math500'  \
	--max_output_tokens=2048 \
	--num_gpus=1 \
	--reward_function='eval_answer_thinking_reward' \
	--tag False > output/benchmark_sampling.log 2>&1

CUDA_VISIBLE_DEVICES=0 python ./src/x_r1/benchmark.py \
	--model_name='Qwen/Qwen2.5-3B-Instruct' \
    --dataset_name='src/fol_r1/gsc' \
	--output_name='./output/result_benchmark_gsc'  \
	--max_output_tokens=2048 \
	--num_gpus=1 \
	--reward_function='eval_answer_reward' \
	--tag False > output/benchmark_sampling.log 2>&1

# data

python src/fol_r1/gsc/generate.py --num_of_var 5 --num_of_generation 3 --num_of_questions 16000