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
'''https://github.com/dhcode-cpp/X-R1'''
'''modify to print online sampling string'''

import os
import textwrap
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union
from unittest.mock import patch

import torch
import torch.utils.data
import torch.nn as nn
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from trl.trainer import GRPOTrainer
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.import_utils import is_vllm_available
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import  pad, selective_log_softmax
from x_grpo_trainer import XGRPOTrainer

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb

import random


# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class XGRPOPlusTrainer(XGRPOTrainer):
    # base GRPO_plus_trainer

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_ps(self, model, input_ids, attention_mask, logits_to_keep):
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        return torch.softmax(logits, dim=-1) # compute probs for the input tokens

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_ps_and_kl_divergence(self, model, input_ids, attention_mask, logits_to_keep):
        # ref_model per_token_probs
        with torch.inference_mode():
            # 如果是peft, base参数共享
            # base 作为 ref model
            # base + Lora 作为 policy model
            if self.ref_model is not None:
                print('is not peft')
                ref_per_token_ps = self._get_per_token_ps(
                    self.ref_model, input_ids, attention_mask, logits_to_keep
                )
            else:
                print('is peft')
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_ps = self._get_per_token_ps(
                        self.model, input_ids, attention_mask, logits_to_keep
                    )

        # model per_token_probs
        model_per_token_ps = self._get_per_token_ps(model, input_ids, attention_mask, logits_to_keep)
        model_per_token_ps_detach = model_per_token_ps.detach()
        
        model_per_token_ps_detach = model_per_token_ps_detach.clamp(min=1e-9)
        ref_per_token_ps = ref_per_token_ps.clamp(min=1e-9)
        model_per_token_ps_detach = model_per_token_ps_detach / model_per_token_ps_detach.sum(dim=-1, keepdim=True)
        ref_per_token_ps = ref_per_token_ps / ref_per_token_ps.sum(dim=-1, keepdim=True)

        # Calculate the KL divergence
        input_ids = input_ids[:, -logits_to_keep:]
        kl_divergence = torch.sum(model_per_token_ps_detach * torch.log(model_per_token_ps_detach/ ref_per_token_ps), dim=-1)
        return kl_divergence, torch.gather(model_per_token_ps, -1, input_ids.unsqueeze(-1)).squeeze(-1)

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        # prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_inputs = Trainer._prepare_inputs(self, inputs = prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm() # 对于PEFT先merge 参数，更新到vllm，
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False)
                if self.quick_eval_dataset is not None and self.state.global_step % self.args.eval_steps == 1:
                    self.run_quick_eval = True
                # perform quick eval with the quick eval dataset if step is a multiple of eval_steps
                if self.quick_eval_dataset is not None and self.state.global_step % self.args.eval_steps == 0 and self.run_quick_eval:
                    quick_eval_outputs = self.llm.generate(
                        [x["prompt"] for x in self.quick_eval_dataset],
                        sampling_params=self.sampling_params,
                        use_tqdm=False,
                    )
                    quick_eval_completion_ids = [out.token_ids for completions in quick_eval_outputs for out in completions.outputs]
                    quick_eval_output_strings = self.processing_class.batch_decode(quick_eval_completion_ids, skip_special_tokens=True)
                    print('-'*100)
                    print('One response from test dataset:', quick_eval_output_strings[random.randint(0, len(quick_eval_output_strings)-1)])
                    print('-'*100)
                    quick_eval_completions = []
                    for completion in quick_eval_output_strings:
                        quick_eval_completions.append([{"role": "assistant", "content": completion}])
                    # calculate the reward for quick_eval_outputs
                    quick_eval_rewards = torch.zeros(len(self.quick_eval_dataset), len(self.reward_funcs), device=device)
                    for i, reward_func in enumerate(self.reward_funcs):
                        if isinstance(reward_func, nn.Module):
                            print('Ignore the reward model')
                        else:
                            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                            reward_kwargs = {key: [example[key] for example in self.quick_eval_dataset] for key in keys}
                            output_reward_func, steps_final_pos = reward_func(completions=quick_eval_completions, regex = self.regex, silence = True, **reward_kwargs)
                            quick_eval_rewards[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
                            # write the reward to the metrics
                            self._metrics[f"quick_eval_rewards/{reward_func.__name__}"].append(quick_eval_rewards[:, i].mean().item())
                    self.run_quick_eval = False

                completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
                for output in outputs:
                    print('-'*100)
                    print('\n\n\n')
                    prompt = output.prompt
                    for output_t in  output.outputs:
                        # print(completion_ids)
                        print('='*100)
                        generated_text = output_t.text
                        print("【USER】: ", prompt )
                        print("\n【ASSISTANT】:", generated_text)
            else:
                completion_ids = [None] * len(all_prompts_text)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_ids = prompt_ids.to(device)
            prompt_mask = prompt_mask.to(device)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            prompt_ids = prompt_ids.to(device)
            prompt_mask = prompt_mask.to(device)
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]


            prompt_string = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
            output_string = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            for prompt, completion in zip(prompt_string, output_string):
                print('='*100)
                print("【USER】: ", prompt )
                print("\n【ASSISTANT】:", completion)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        completion_mask_adjustment = []
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = Trainer._prepare_inputs(self, inputs = reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func, steps_final_pos = reward_func(prompts=prompts, completions=completions, regex = self.regex, **reward_kwargs)
                if len(steps_final_pos) > 0 and self.part_of_gradient:
                    completion_mask_adjustment = steps_final_pos
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        # print('x_grpo_rewars output:',rewards)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Adjust the completion mask if needed
        if self.part_of_gradient:
            for i in range(len(completion_mask_adjustment)):
                if completion_mask_adjustment[i] != 0 and advantages[i] < 0:
                    text = completions_text[i]
                    char_pos = completion_mask_adjustment[i]
                    # Re-tokenize
                    encoding = self.processing_class(text, return_offsets_mapping=True, add_special_tokens=False)
                    offsets = encoding["offset_mapping"]
                    target_token_idx = -1
                    for token_idx, (start, end) in enumerate(offsets):
                        # Find the first token that *ends* after the target character position.
                        if end > char_pos:
                            target_token_idx = token_idx
                            break
                    completion_mask[i, :target_token_idx + 1] = 0

        # Log the metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
        ):
            import pandas as pd

            # For logging
            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "reward": rewards.tolist(),
            }
            df = pd.DataFrame(table)

            if wandb.run is not None and self.accelerator.is_main_process:
                wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
        }


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_kl_per_func, per_token_ps = self._get_per_token_ps_and_kl_divergence(model, input_ids, attention_mask, logits_to_keep)
        per_token_kl_mean = torch.mean(gather(torch.mean(per_token_kl_per_func)))
        
        per_token_kl = per_token_kl_per_func - per_token_kl_mean
        # x - x.detach() allows for preserving gradients from x
        advantages = inputs["advantages"]
        per_token_loss = - (per_token_ps / per_token_ps.detach()) * (advantages.unsqueeze(1) - self.beta * per_token_kl)
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        # loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        self._metrics["kl"].append(per_token_kl_mean)

        return loss
