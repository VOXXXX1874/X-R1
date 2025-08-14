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

from typing import Any, Callable, Optional, Sized, Union
import torch
import torch.nn as nn
from accelerate.utils import broadcast_object_list, gather, gather_object
from datasets import Dataset, IterableDataset
from torch import nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.utils import is_peft_available
from trl.trainer import GRPOTrainer
from trl.extras.profiling import profiling_decorator
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.import_utils import is_vllm_available
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import  pad, selective_log_softmax

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb

import random

from utils.MDP import *
import json


# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

class XGRPOTrainerMDP(GRPOTrainer):
    # base trl GRPO_trainer


    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        quick_eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
    ):
        # Read and parse the MDP tree according to the question ID.
        self.tree_dict = {}
        for item in train_dataset:
            self.tree_dict[item['id']] = MDP_tree_from_string(item['MDP_tree'])
        # Save the trees every time all the trees are updated 
        self.last_tree_save = 0

        GRPOTrainer.__init__(
            self,
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )

        # quick eval dataset
        if quick_eval_dataset is not None:
            def make_prompt(example):
                example['prompt'] = apply_chat_template(example, self.processing_class)["prompt"]
                return example
            
            self.quick_eval_dataset = quick_eval_dataset.map(make_prompt)
            self.run_quick_eval = False
        else:
            self.quick_eval_dataset = None
        # Modify the sampling parameters
        if args.extra_generations > 0:
            self.sampling_params = SamplingParams(
                    temperature=args.temperature,
                    max_tokens=self.max_completion_length,
                    guided_decoding=None,
                    n=args.extra_generations,
                )
        # Determine whether to use tag in response
        self.tag = args.tag
        # Determine number of questions per step
        self.num_questions_per_batch = self.args.per_device_train_batch_size * self.accelerator.num_processes // self.num_generations
        if self.num_questions_per_batch * args.extra_generations % self.accelerator.num_processes != 0:
            raise ValueError(
                f"Number of prompts ({self.num_questions_per_batch}) times extra generations ({args.extra_generations}) "
                f"must be divisible by the number of processes ({self.accelerator.num_processes})."
            )
        if self.num_questions_per_batch > self.accelerator.num_processes:
            raise ValueError(
                f"Number of questions ({self.num_questions_per_batch}) per step must be less than or equal to the number of processes ({self.accelerator.num_processes})."
            )
        if args.extra_generations % self.num_generations != 0:
            raise ValueError(
                f"Number of extra generations ({args.extra_generations}) must be divisible by the number of generations ({self.num_generations})."
            )

    @profiling_decorator
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        # Sample index uniformly
        indices = torch.multinomial(torch.ones((inputs['advantages_list'].size()[0],)), num_samples=self.args.per_device_train_batch_size, replacement=False)
        # Select the inputs based on the sampled indices
        inputs = {
            key: value[indices] if isinstance(value, torch.Tensor) else [value[i] for i in indices]
            for key, value in inputs.items()
        }
        return inputs

    def _generate_and_score_completions(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        inputs = inputs * (self.args.extra_generations // self.num_generations)
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
                self._move_model_to_vllm() 
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                # Check if we need to save the tree
                if (self.state.global_step/self.num_iterations) * self.num_questions_per_batch * self.args.gradient_accumulation_steps // len(self.train_dataset) > self.last_tree_save:
                    self.last_tree_save = (self.state.global_step/self.num_iterations) * self.num_questions_per_batch * self.args.gradient_accumulation_steps // len(self.train_dataset)
                    tree_dict_to_save = []
                    for item in self.train_dataset:
                        item['MDP_tree'] = self.tree_dict[item['id']].__repr__()
                        tree_dict_to_save.append(item)
                    # Open a json file in self.args.output_dir and save the tree_dict
                    with open(f"{self.args.output_dir}/tree_dict_{self.last_tree_save}.json", "w") as f:
                        json.dump(tree_dict_to_save, f)

                # Check if we need to run quick eval
                if self.quick_eval_dataset is not None and (self.state.global_step/self.num_iterations) % self.args.eval_steps == 1:
                    self.run_quick_eval = True
                # perform quick eval with the quick eval dataset if step is a multiple of eval_steps
                if (self.quick_eval_dataset is not None and (self.state.global_step/self.num_iterations) % self.args.eval_steps == 0 and self.run_quick_eval) or self.state.global_step == 0:
                    quick_eval_outputs = self.llm.generate(
                        [x["prompt"] for x in self.quick_eval_dataset],
                        sampling_params=SamplingParams(
                            temperature=self.sampling_params.temperature,
                            max_tokens=self.max_completion_length,
                            guided_decoding=self.sampling_params.guided_decoding,
                            n=1,
                        ),
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
                            # FIXME: Compatible with different train and eval dataset
                            keys = [key for key in self.quick_eval_dataset[0] if key not in ["prompt", "completion"]]
                            reward_kwargs = {key: [example[key] for example in self.quick_eval_dataset] for key in keys}
                            output_reward_func, steps_final_pos = reward_func(completions=quick_eval_completions, tag = self.tag, silence = True, **reward_kwargs)
                            quick_eval_rewards[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
                            # write the reward to the metrics
                            self._metrics['train'][f"quick_eval_rewards/{reward_func.__name__}"].append(quick_eval_rewards[:, i].mean().item())
                    self.run_quick_eval = False

                ordered_set_of_prompts = list(dict.fromkeys(all_prompts_text))
                
                outputs = self.llm.generate(
                    ordered_set_of_prompts, sampling_params=self.sampling_params, use_tqdm=False
                )

                completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
            else:
                completion_ids = [None] * len(all_prompts_text)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            # With extra generations, we need to slice the completions to match the actual number of generations.
            per_process_partition = self.num_questions_per_batch * self.args.extra_generations // self.accelerator.num_processes
            process_slice = slice(
                self.accelerator.process_index * per_process_partition,
                (self.accelerator.process_index + 1) * per_process_partition,
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_ids = prompt_ids.to(device)
            prompt_mask = prompt_mask.to(device)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            raise ValueError(
                "Only vLLM is supported for now. Please set `args.use_vllm` to `True`."
            )

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                # Because there are extra generations, we need to divide prompt_completion_ids into batches
                # to avoid OOM. 
                old_per_token_logps = []
                batch_size = self.args.per_device_train_batch_size * self.accelerator.num_processes
                for i in range(0, prompt_completion_ids.size(0), batch_size):
                    batch_prompt_completion_ids = prompt_completion_ids[i : i + batch_size]
                    batch_attention_mask = attention_mask[i : i + batch_size]
                    batch_old_per_token_logps = self._get_per_token_logps(
                        self.model, batch_prompt_completion_ids, batch_attention_mask, logits_to_keep
                    )
                    old_per_token_logps.append(batch_old_per_token_logps)
                old_per_token_logps = torch.cat(old_per_token_logps, dim=0)
                # Move to device
                old_per_token_logps = old_per_token_logps.to(device)
            else:
                old_per_token_logps = None

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
                with torch.no_grad():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func, _ = reward_func(completions=completions, tag = self.tag, silence = True, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        if self.tag:
            # For the format reward, we follow the implementation of GRPO algo
            format_rewards = gather(rewards_per_func)[:,1]

            # Compute grouped-wise rewards
            mean_grouped_format_rewards = format_rewards.view(-1, self.args.extra_generations).mean(dim=1)
            std_grouped_format_rewards = format_rewards.view(-1, self.args.extra_generations).std(dim=1)

            # Normalize the rewards to compute the advantages
            mean_grouped_format_rewards = mean_grouped_format_rewards.repeat_interleave(self.args.extra_generations, dim=0)
            std_grouped_format_rewards = std_grouped_format_rewards.repeat_interleave(self.args.extra_generations, dim=0)
            format_advantages_list = (format_rewards - mean_grouped_format_rewards) / (std_grouped_format_rewards + 1e-4)

            # Slice to keep only the local part of the data
            format_advantages_list = format_advantages_list[process_slice]

        # If no tag, we can parse the completions_text and use reward directly
        question_id = inputs[0]["id"]
        expressions_rewards_pair = []
        for i, completion in enumerate(completions_text):
            # For the response that is not well-formed, we skip it
            if self.tag and rewards_per_func[i,1] < 0.1:
                expressions_rewards_pair.append((-1, [], [], 0))
                continue
            # Extract the number from the expressions in completions text
            expressions, positions = thinking_parse(
                completion,
                extraction_config=[LatexExtractionConfig()],
            )
            # Convert the character positions to token positions
            # Re-tokenize and get the offsets_mapping
            encoding = self.processing_class(completion, return_offsets_mapping=True, add_special_tokens=False)
            offsets = encoding["offset_mapping"]
            position_pointer = 0
            for token_idx, (_, end) in enumerate(offsets):
                # Find the corresponding character position in the completion
                while position_pointer < len(positions) -1 and positions[position_pointer] < end:
                    positions[position_pointer] = token_idx
                    position_pointer += 1
            expressions_rewards_pair.append((question_id, expressions, positions, rewards_per_func[i][0].item()))
        # Gather the expressions_rewards_pair and update the MDP tree
        all_expressions_rewards_pair = gather_object(expressions_rewards_pair)
        past_id = -1
        for i, (id, expressions, positions, reward) in enumerate(all_expressions_rewards_pair):
            if self.tag and id == -1:
                continue
            if id != past_id:
                self.tree_dict[id].bfs_decay(self.args.delta)
                past_id = id
            if id not in self.tree_dict:
                raise ValueError(f"Question ID {id} not found in the MDP tree.")
            # Update the MDP tree with the new reward
            self.tree_dict[id].update(expressions, reward)
        # Update the state value in the MDP tree
        self.tree_dict[question_id].update_node_value()
        # According to the updated MDP tree, allocate the advantages and calculate the probability for each generation
        advantages_list = []
        for i, expressions_reward in enumerate(expressions_rewards_pair):
            if self.tag and rewards_per_func[i,1] < 0.1:
                # If the generation is not well-formed, we only consider the format advantages
                advantages = torch.zeros(completion_ids.size(1), device=device)
            else:
                id, expressions, positions, reward = expressions_reward
                advantages = self.tree_dict[id].advantages(
                    expressions, positions, completion_ids.size(1), final_reward=reward
                )
                advantages = torch.tensor(advantages, device=device)
            advantages = advantages * self.reward_weights[0].to(device)
            if self.tag:
                # Use reward_weights to weighted sum the advantages and format advantages
                advantages += format_advantages_list[i] * self.reward_weights[1].to(device)

            advantages_list.append(advantages)

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        reward_per_func = gather(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[mode][f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            # Log the completions, advantages distribution in decoded token, and reward
            if self.accelerator.is_main_process:
                for i, advantages in enumerate(advantages_list):
                    print('*' * 100)
                    print(f"Prompt: {prompts_text[i]}")
                    print('-' * 100)
                    last_advantage_pos = 0
                    for j, advantage in enumerate(advantages):
                        if advantage != advantages[last_advantage_pos]:
                            print(f"In the {last_advantage_pos} to {j} tokens, the advantage is {advantages[last_advantage_pos]}.")
                            decoded_segment = self.processing_class.decode(
                                completion_ids[i][last_advantage_pos:j], skip_special_tokens=True
                            )
                            print("In that position, the decoded completion is:")
                            print(decoded_segment)
                            print('-' * 100)
                            last_advantage_pos = j
                    print(f"In the {last_advantage_pos} to {len(advantages)} position, the advantage is {advantages[last_advantage_pos]}.")
                    decoded_segment = self.processing_class.decode(
                        completion_ids[i][last_advantage_pos:], skip_special_tokens=True
                    )
                    print(f"In that position, the decoded completion is: {decoded_segment}")
                    print('-' * 100)
                    print(f"The accuracy reward for this completion is {rewards_per_func[i][0].item()}.")
                    if self.tag:
                        print(f"The format reward for this completion is {rewards_per_func[i][1].item()}.")
                    print('*' * 100)

        advantages_list = torch.stack(advantages_list).to(device=device, dtype=torch.bfloat16)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "advantages_list": advantages_list,
        }
    
    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # Compute the per-token log probabilities for the reference model
        with torch.no_grad():
            # Because there is extra generations, the reference model is moved here to save computation.
            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages_list"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss
