# 1 3060 6GB
ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero1.yaml \
--num_processes=1 \
src/x_r1/grpo.py \
--config recipes/X_R1_0dot5B_experimental_config.yaml \
> ./output/x_r1_experimental_sampling.log 2>&1

# 2 3090
ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
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
--num_processes=6 src/x_r1/grpo.py \
--config recipes/X_R1_1dot5B_config_gsc.yaml \
> ./output/x_r1_1dot5B_sampling.log 2>&1

ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=6 src/x_r1/grpo.py \
--config recipes/X_R1_1dot5B_config_gsc.yaml \
> ./output/x_r1_1dot5B_sampling-1.log 2>&1

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

# num

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=2 \
src/x_r1/grpo.py \
--config recipes/X_R1_1dot5B_config_math_num.yaml \
> ./output/X_R1_1dot5B_math_num.log 2>&1

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=4 \
src/x_r1/grpo.py \
--config recipes/X_R1_1dot5B_config_math_num.yaml \
> ./output/X_R1_1dot5B_math_num.log 2>&1

# MDP

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=1 \
src/x_r1/mdp.py \
--config recipes/MDP1_0dot5B_experimental_config.yaml \
> ./output/MDP1_experimental_sampling.log 2>&1

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=3 \
src/x_r1/mdp.py \
--config recipes/MDP1_1dot5B_config.yaml \
> ./output/MDP1_1dot5B_sampling.log 2>&1