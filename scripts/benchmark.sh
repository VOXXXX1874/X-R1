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

CUDA_VISIBLE_DEVICES=0 python ./src/x_r1/benchmark.py \
	--model_name='records/X-R1-1.5B_gsc_g4_2048_cv' \
    --dataset_name='src/fol_r1/gsc' \
	--output_name='./output/result_benchmark_gsc'  \
	--max_output_tokens=2048 \
	--num_gpus=1 \
	--reward_function='eval_answer_reward' \
	--tag True > output/benchmark_sampling.log 2>&1

CUDA_VISIBLE_DEVICES=0 python ./src/x_r1/benchmark.py \
	--model_name='Qwen/Qwen2.5-3B-Instruct' \
    --dataset_name='src/cv_extraction/MATH-500/exp' \
	--output_name='./output/result_benchmark_math500'  \
	--max_output_tokens=4096 \
	--num_gpus=1 \
	--reward_function='eval_answer_thinking_reward' \
	--tag False \
	--regex True > output/benchmark_sampling.log 2>&1

CUDA_VISIBLE_DEVICES=0 python ./src/x_r1/benchmark_MVOT_rule.py \
	--model_name='./records/Qwen2.5-1.5B-MVOT' \
    --dataset_name='src/cv_extraction/MATH-500/exp' \
	--output_name='./output/result_benchmark_math500'  \
	--max_output_tokens=4096 \
	--num_generation=1 \
	--max_steps=0 \
	--num_gpus=1 > output/benchmark_sampling.log 2>&1

CUDA_VISIBLE_DEVICES=0 python ./src/x_r1/benchmark.py \
	--model_name='Qwen/Qwen2.5-1.5B-Instruct' \
    --dataset_name='src/cv_extraction/MATH-500/exp' \
	--output_name='./output/result_benchmark_math500'  \
	--max_output_tokens=4096 \
	--num_gpus=1 \
	--reward_function='eval_answer_reward' \
	--tag False

CUDA_VISIBLE_DEVICES=0 python ./src/x_r1/benchmark_MVOT_rule.py \
	--model_name='./records/Qwen2.5-1.5B-MVOT' \
    --dataset_name='src/cv_extraction/MATH-500/exp' \
	--output_name='./output/result_benchmark_math500'  \
	--max_output_tokens=4096 \
	--num_generation=50 \
	--max_steps=3 \
	--num_gpus=1 > output/benchmark_sampling.log 2>&1

CUDA_VISIBLE_DEVICES=0 python ./src/x_r1/benchmark_MVOT_latent.py \
	--model_name='./records/Qwen2.5-1.5B-MVOT' \
    --dataset_name='src/cv_extraction/MATH-500/exp' \
	--output_name='./output/result_benchmark_math500'  \
	--max_output_tokens=1024 \
	--num_generation=32 \
	--max_steps=8 \
	--num_gpus=1 > output/benchmark_sampling.log 2>&1

CUDA_VISIBLE_DEVICES=0 python ./src/x_r1/benchmark_MVOT_latent.py \
	--model_name='./records/Qwen2.5-1.5B-MVOT' \
    --dataset_name='src/cv_extraction/MATH-500/exp' \
	--output_name='./output/result_benchmark_math500'  \
	--max_output_tokens=1024 \
	--num_generation=32 \
	--max_steps=8 \
	--num_gpus=1 \
	--layer=-2 \
	--temperature=1.0 > output/benchmark_sampling.log 2>&1

CUDA_VISIBLE_DEVICES=0 python ./src/x_r1/benchmark_MVOT_latent_pencil.py \
	--model_name='./records/Qwen2.5-1.5B-pencil-xr1' \
    --dataset_name='src/cv_extraction/MATH-500/exp' \
	--output_name='./output/result_benchmark_math500'  \
	--max_output_tokens=1024 \
	--num_generation=1 \
	--max_steps=0 \
	--num_gpus=1 \
	--layer=-1 \
	--temperature=0.7 > output/benchmark_sampling.log 2>&1

CUDA_VISIBLE_DEVICES=0 python ./src/x_r1/benchmark_MVOT_latent_pencil.py \
	--model_name='./records/Qwen2.5-1.5B-pencil-xr1' \
    --dataset_name='src/cv_extraction/MATH-500/exp' \
	--output_name='./output/result_benchmark_math500'  \
	--max_output_tokens=1024 \
	--num_generation=1 \
	--max_steps=0 \
	--num_gpus=1 \
	--layer=-1 \
	--temperature=0.7 \
	--num_samples=1 > output/benchmark_sampling.log 2>&1

CUDA_VISIBLE_DEVICES=0 python ./src/x_r1/benchmark.py \
	--model_name='Qwen/Qwen2.5-1.5B-Instruct' \
    --dataset_name='src/cv_extraction/MATH-500/exp_num' \
	--output_name='./output/result_benchmark_math500'  \
	--max_output_tokens=4096 \
	--num_gpus=1 \
	--reward_function='eval_answer_thinking_reward' \
	--tag False \
	--reward_type 'num' > output/benchmark_sampling.log 2>&1