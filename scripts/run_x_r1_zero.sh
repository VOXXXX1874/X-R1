
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