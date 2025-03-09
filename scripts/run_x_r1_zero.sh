
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

# 1 3090/4090
ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero1.yaml \
--num_processes=1 \
src/x_r1/grpo.py \
--config recipes/X_R1_zero_0dot5B_peft_config.yaml \
> ./output/x_r1_test_sampling.log 2>&1

# 2 3090/4090 ???
ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=2 src/x_r1/grpo.py \
--config recipes/X_R1_zero_0dot5B_peft_config.yaml \
> ./output/x_r1_test_sampling.log 2>&1


# 4 3090/4090 projgpu8
# remember export LD_LIBRARY_PATH=/research/d2/spc/zzchen2/anaconda/envs/xr1/lib:$LD_LIBRARY_PATH
ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3_offparam.yaml \
--num_processes=3 src/x_r1/grpo.py \
--config recipes/X_R1_zero_3B_config_low_memory.yaml \
> ./output/x_r1_3B_sampling.log 2>&1

ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=3 src/x_r1/grpo.py \
--config recipes/X_R1_zero_3B_config.yaml \
> ./output/x_r1_3B_sampling.log 2>&1

ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=3 src/x_r1/grpo.py \
--config recipes/X_R1_3B_config.yaml \
> ./output/x_r1_3B_sampling.log 2>&1

ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=3 src/x_r1/grpo.py \
--config recipes/X_R1_3B_supervised_0dot5B_config.yaml \
> ./output/X_R1_3B_supervised_0dot5B.log 2>&1

ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=3 src/x_r1/grpo.py \
--config recipes/X_R1_3B_supervised_1dot5B_config.yaml \
> ./output/X_R1_3B_supervised_1dot5B.log 2>&1

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./src/x_r1/benchmark.py \
	--model_name='records/X-R1-3B_0' \
    --dataset_name='HuggingFaceH4/MATH-500' \
	--output_name='./output/result_benchmark_math500'  \
	--max_output_tokens=1024 \
	--num_gpus=4

CUDA_VISIBLE_DEVICES=0 python ./src/x_r1/benchmark.py \
	--model_name='records/X-R1-3B-7500-epoch1' \
    --dataset_name='HuggingFaceH4/MATH-500' \
	--output_name='./output/result_benchmark_math500'  \
	--max_output_tokens=1024 \
	--num_gpus=1