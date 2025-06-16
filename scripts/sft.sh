# One 1 node of 8 x H100s
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name open-r1/OpenR1-Math-220k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill

# 2 3090
ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=2 src/x_r1/sft.py \
--config recipes/SFT_0dot5B_experimental_config.yaml \
> ./output/sft_experimental_sampling.log 2>&1

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=2 src/x_r1/sft.py \
--config recipes/SFT_1dot5B_config.yaml \
> ./output/sft_1dot5B_sampling.log 2>&1

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=2 src/x_r1/sft.py \
--config recipes/Pencil_1dot5B_config.yaml \
> ./output/pencil_1dot5B_sampling.log 2>&1