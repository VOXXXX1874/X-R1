# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from dataclasses import dataclass, field

import datasets
import torch
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer


from configs import GRPOConfig, GRPOScriptArguments
from rewards import (
    accuracy_thinking_reward,
    accuracy_reward,
    thinking_reward,
    format_reward
)
from utils.callbacks import get_callbacks
import utils.prepare_dataset
from utils.prepare_dataset import prepare_dataset, prepare_quick_eval_dataset, SYSTEM_PROMPT_TAG
from utils.wandb_logging import init_wandb_training
from x_grpo_trainer import XGRPOTrainer
from x_grpo_plus_trainer import XGRPOPlusTrainer
from x_grpo_supervised_trainer import XGRPOSupervisedTrainer
from trl import ModelConfig, TrlParser, get_peft_config
from peft import LoraConfig, PeftModel, get_peft_model


logger = logging.getLogger(__name__)


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset
    dataset = prepare_dataset(script_args.dataset_name, script_args.dataset_train_split, training_args.tag)

    if script_args.quick_eval_dataset:
        quick_eval_dataset = prepare_quick_eval_dataset(script_args.quick_eval_dataset, training_args.tag)

    # Get reward functions
    REWARD_FUNCS_REGISTRY = {
        "accuracy_thinking": accuracy_thinking_reward,
        "accuracy": accuracy_reward,
        "thinking": thinking_reward,
        "format": format_reward
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )

    training_args.gradient_checkpointing = False
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )

    # model = AutoModelForCausalLM.from_pretrained(**model_kwargs, pretrained_model_name_or_path = model_args.model_name_or_path)
    training_args.model_init_kwargs = model_kwargs
    # peft_config=get_peft_config(model_args)
    # print(peft_config)
    # if peft_config not None:
    #     model = get_peft_model(model, peft_config)
    # print(model)


    #############################
    # Initialize the XGRPO trainer
    #############################
    if script_args.trainer_type == "XGRPOTrainer":
        trainer = XGRPOTrainer(
            model=model_args.model_name_or_path,
            # model = model,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
            peft_config=get_peft_config(model_args), # LoRA parameter
            callbacks=get_callbacks(training_args, model_args),
            quick_eval_dataset=quick_eval_dataset if script_args.quick_eval_dataset else None,
        )
    elif script_args.trainer_type == "XGRPOPlusTrainer":
        trainer = XGRPOPlusTrainer(
            model=model_args.model_name_or_path,
            # model = model,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
            peft_config=get_peft_config(model_args), # LoRA parameter
            callbacks=get_callbacks(training_args, model_args),
            quick_eval_dataset=quick_eval_dataset if script_args.quick_eval_dataset else None,
        )
    elif script_args.trainer_type == "XGRPOSupervisedTrainer":
        trainer = XGRPOSupervisedTrainer(
            model=model_args.model_name_or_path,
            reference_model=script_args.reference_model,
            # model = model,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
            peft_config=get_peft_config(model_args), # LoRA parameter
            callbacks=get_callbacks(training_args, model_args),
            quick_eval_dataset=quick_eval_dataset if script_args.quick_eval_dataset else None,
        )
    else:
        raise ValueError(f"Invalid trainer type: {script_args.trainer_type}")

    print(trainer)

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["X-R1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #     metrics = trainer.evaluate()
    #     metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args )
