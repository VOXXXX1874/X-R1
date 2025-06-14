from datasets import load_dataset
from vllm import LLM, SamplingParams
import argparse
import json
from rewards import eval_answer_reward
# import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def eval_thinking_reward(completion, gold_answer, gold_process, silence = False):
    raise NotImplementedError('eval_thinking_reward is not implemented yet')

def create_dataset(dataset_name, tokenizer):
    dataset = load_dataset(dataset_name, split='test')

    def make_conversation(example):
        return {
            "prompt": [
                {"role": "user", "content": example["problem"]},
            ],
        }

    dataset = dataset.map(make_conversation)

    def make_latex(example):
        example["answer"] = '$' + str(example["answer"]) + '$'
        return example

    dataset = dataset.map(make_latex)

    #def format_function(example):
    #    example['prompt'] = tokenizer.apply_chat_template(example['prompt'], tokenize = False, add_generation_prompt = True )
    #    return example
    #
    #dataset = dataset.map(format_function, batched = False)
        
    return dataset

def mean_simcount_cosine_similarity(hidden_states):
    """
    Calculate the mean of the hidden states and return the cosine similarity.
    """
    mean_hidden_states = torch.stack([torch.mean(hidden_states[i], dim=0) for i in range(len(hidden_states))])
    # Normalize the mean hidden states
    mean_hidden_states = mean_hidden_states / mean_hidden_states.norm(dim=-1, keepdim=True)
    # Calculate the cosine similarity between the hidden states
    cosine_similarities = mean_hidden_states @ mean_hidden_states.T
    # Sum the cosine similarities for each intermediate result
    summed_similarities = (cosine_similarities > 0.975).sum(dim=1)
    # Find the index of the maximum similarity
    max_index = summed_similarities.argmax().item()
    # Return the index of the intermediate result with the maximum similarity
    return max_index

def mean_naive_cosine_similarity(hidden_states):
    """
    Calculate the mean of the hidden states and count the number of vectors that have a cosine similarity above a threshold.
    """
    mean_hidden_states = torch.stack([torch.mean(hidden_states[i], dim=0) for i in range(len(hidden_states))])
    # Normalize the mean hidden states
    mean_hidden_states = mean_hidden_states / mean_hidden_states.norm(dim=-1, keepdim=True)
    # Calculate the cosine similarity between the hidden states
    cosine_similarities = mean_hidden_states @ mean_hidden_states.T
    # Sum the cosine similarities for each intermediate result
    summed_similarities = cosine_similarities.sum(dim=1)
    # Find the index of the maximum similarity
    max_index = summed_similarities.argmax().item()
    # Return the index of the intermediate result with the maximum similarity
    return max_index

def intermediate_majority_voting(model, tokenizer, selected_context, intermediate_results, layer, batch_size=2):
    """
    Perform majority voting on the intermediate results.
    """
    # recover the full context
    if selected_context[-1]['role'] == 'user':
        full_context_list = [[{"role": "user", "content": selected_context[-1]["content"]}, {"role": "assistant", "content": intermediate_results[i]}] for i in range(len(intermediate_results))]
    elif selected_context[-1]['role'] == 'assistant':
        full_context_list = [[{"role": "user", "content": selected_context[-2]["content"]}, {"role": "assistant", "content": selected_context[-1]["content"] + intermediate_results[i]}] for i in range(len(intermediate_results))]
    # Calculate the hidden states for the full context
    full_context_list = [tokenizer.apply_chat_template(context, tokenize=False, continue_final_message=True) for context in full_context_list]
    
    with torch.inference_mode():
        all_hidden_states = []
        # To avoid OOM, we inference 8 contexts at a time
        for i in range(0, len(full_context_list), batch_size):
            # also return the offset mapping to find the position of <record> or <answer> tag
            model_inputs = tokenizer(full_context_list[i:i+batch_size], return_tensors="pt", padding=True, return_offsets_mapping=True).to(model.device)
            attention_mask_count_list = model_inputs["attention_mask"].shape[1] - model_inputs["attention_mask"].sum(dim=1)
            offsets_list = model_inputs.pop("offset_mapping")
            # Find the character position of <record> or <answer> tag
            char_pos_list = []
            for j, offsets in enumerate(offsets_list):
                char_pos = full_context_list[i + j].rfind("<answer>")
                if char_pos == -1:
                    char_pos = full_context_list[i + j].rfind("<record>")
                print("context after <answer> or <record> tag: ", full_context_list[i + j][char_pos:])
                for token_idx, (start, end) in enumerate(offsets):
                    # Find the first token that *ends* after the target character position.
                    if end > char_pos:
                        char_pos = token_idx
                        break
                char_pos_list.append(char_pos)
            # get the hidden states for the current batch
            layer_hidden_states = model(**model_inputs, output_hidden_states=True)["hidden_states"][layer]
            # only preserve the hidden states for the tokens after the <answer> or <final> tag
            for hidden_state, char_pos, attention_mask_count in zip(layer_hidden_states, char_pos_list, attention_mask_count_list):
                selected_hidden_state = hidden_state[char_pos:-attention_mask_count, :] if attention_mask_count > 0 else hidden_state[char_pos:, :]
                # Append the hidden states to the list
                all_hidden_states.append(selected_hidden_state)

    # Perform majority voting on the hidden states
    # TODO: A better way to do majority voting
    # Here we use the mean of the hidden states as the representative
    return mean_naive_cosine_similarity(all_hidden_states)

def vllm_generate(model_name, output_name, dataset_name, num_gpus, max_output_tokens):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # evaluation dataset
    dataset = create_dataset(dataset_name, tokenizer)
    print(dataset)


    answers = []
    prompts = []
    processes = []
    count = 0
    for data in dataset:
        answers.append(data['answer'])
        prompts.append(data['prompt'])
        processes.append(data['process']) if 'process' in data else processes.append('')
        #count += 1
        #if count == 10:
        #    break
    # Create LLM object
    llm = LLM(model=model_name,  # replace your own model
                dtype='bfloat16',
                tensor_parallel_size=num_gpus,  # number of gpu
                gpu_memory_utilization=0.9,  # prevent OOM
                trust_remote_code=True,
                # use_cache=False,
              )
    
    # Create transformers LLM for hidden states
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    sampling_params = SamplingParams(temperature=args.temperature,
                            max_tokens=max_output_tokens,
                            )

    pro_scores = []
    acc_scores = []
    result_all = []
    total_acc = 0
    total_format = 0
    for prompt, gold_answer, gold_process in zip (prompts, answers, processes):
        selected_context = prompt
        # Create a sampling params for regular steps
        print('selected context: ', selected_context)
        for i in range(args.max_steps + 1):
            # If there is <answer> tag in the context, we stop the generation
            if selected_context[-1]["content"].find("<answer>") != -1:
                break
            # Tokenize the selected context
            if i == 0:
                input_ids = tokenizer.apply_chat_template(selected_context, tokenize = False, add_generation_prompt = True )
            else:
                input_ids = tokenizer.apply_chat_template(selected_context, tokenize = False, continue_final_message = True )
            # vllm generation
            outputs = llm.generate([input_ids] * args.num_generation,
                                    sampling_params, use_tqdm=False)
            # If this is the max steps, we directly go to the final answer
            if i == args.max_steps:
                intermediate_results = []
                # convert selected context to rollout
                if selected_context[-1]['role'] == 'user':
                    rollout = [selected_context.append({"role": "assistant", "content": output.outputs[0].text}) for output in outputs]
                elif selected_context[-1]['role'] == 'assistant':
                    rollout = [selected_context] * args.num_generation
                    for j, output in enumerate(outputs):
                        rollout[j][-1]["content"] += output.outputs[0].text
                selected_context = selected_context[:-1]  # remove the last message, which is the model output
                # go to final answer
                for j in range(0, 20):
                    # if any of the rollouts contains <answer> tag, we add the rollout to intermediate results and remove it from rollout list
                    remove_indices = []
                    for k, r in enumerate(rollout):
                        if r[-1]["content"].find("<answer>") != -1:
                            intermediate_results.append(r[-1]["content"])
                            remove_indices.append(k)
                        else:
                            # erase the thinking in the rollout
                            rollout[k][-1]["content"] = re.sub(r'<think>.*?</think>', '', rollout[k][-1]["content"], flags=re.DOTALL)
                    # remove the rollouts that contain <answer> tag
                    for k in sorted(remove_indices, reverse=True):
                        del rollout[k]
                    # if there are no rollouts left, break
                    if len(rollout) == 0:
                        break
                    input_ids = tokenizer.apply_chat_template(rollout, tokenize = False, continue_final_message = True )
                    outputs = llm.generate(input_ids,
                                            sampling_params, use_tqdm=False)
                    # append the generated content to rollouts
                    for k, output in enumerate(outputs):
                        rollout[k][-1]["content"] += output.outputs[0].text
                    
            else:
                # collect all the intermediate results
                intermediate_results = []
                for output in outputs:
                    intermediate_results.append(output.outputs[0].text)

            if len(intermediate_results) == 0:
                break

            # Perform majority voting on the intermediate results
            # <answer> is also part of majority voting
            # erase the thinking in selected intermediate result
            selected_intermediate_result = re.sub(r'<think>.*?</think>', '', intermediate_results[intermediate_majority_voting(model, tokenizer, selected_context, intermediate_results, args.layer)], flags=re.DOTALL)
            if selected_context[-1]['role'] == 'user':
                selected_context.append({"role": "assistant", "content": selected_intermediate_result})
            elif selected_context[-1]['role'] == 'assistant':
                selected_context[-1]["content"] += selected_intermediate_result
            print('selected context: ', selected_context)
        
        completion = selected_context[-1]["content"]
        
        if args.reward_function == 'eval_answer_reward':
            acc_score, _ = eval_answer_reward(completion, gold_answer, silence = False)
            pro_score = 0
        elif args.reward_function == 'eval_answer_thinking_reward':
            acc_score, _ = eval_answer_reward(completion, gold_answer, silence = False)
            pro_score, step_end_pos = eval_thinking_reward(completion, gold_answer, gold_process, silence = False)
        acc_scores.append(acc_score)
        pro_scores.append(pro_score)
        total_acc = total_acc + acc_score

        # print('format score', format_score)
        # print('accuracy score', acc_score)
        # print('-'*100)

        result_all.append({
            'prompt': prompt, 
            'completion': completion, 
            'gold answer': gold_answer, 
            'acc scores': acc_score,  
            'pro scores': pro_score,
        })

    print('='*100)
    print('eval acc: ', total_acc / len(acc_scores))
    print('eval pro: ', sum(pro_scores) / len(pro_scores))

    current_result_file = output_name + '.json'
    with open(current_result_file, 'w', encoding='utf-8') as file:
        json.dump(result_all, file, ensure_ascii=False, indent=4)
        
    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_name',  type=str, default='', required=True,
                        help='model name path')
    parser.add_argument('--output_name', type=str, default='', required=True,
                        help='output path')
    parser.add_argument('--dataset_name', type=str, default='HuggingFaceH4/MATH-500', required=True,
                        help='dataset path')
    parser.add_argument('--max_output_tokens', type=int, default=100,
                        help='generation tokens')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='generation tokens')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='generation tokens')
    parser.add_argument('--reward_function', type=str, default='eval_answer_reward',
                        help='reward function')
    parser.add_argument('--num_generation', type=int, default=6,
                        help='number of trials generated for each step')
    parser.add_argument('--max_steps', type=int, default=6,
                        help='maximum number of steps for each problem')
    parser.add_argument('--layer', type=int, default=-1,
                        help='hidden state from which layer to extract, -1 means the last layer')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='temperature for sampling')
    args = parser.parse_args()
    print(args)

    if args.reward_function != 'eval_answer_reward' and args.reward_function != 'eval_answer_thinking_reward':
        raise ValueError('reward function not found')

    vllm_generate(args.model_name,
                  args.output_name,
                  args.dataset_name,
                  args.num_gpus,
                  args.max_output_tokens,)
    # print(f'toxicity score mean: {mean}, toxicity score std: {std}')