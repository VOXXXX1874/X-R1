# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./src/x_r1/benchmark.py \
# 	--model_name='xiaodongguaAIGC/X-R1-3B \
#   --dataset_name='HuggingFaceH4/MATH-500' \
# 	--output_name='./output/result_benchmark_math500'  \
# 	--max_output_tokens=1024 \
# 	--num_gpus=4

# CUDA_VISIBLE_DEVICES=0 python ./src/x_r1/benchmark.py \
# 	--model_name='xiaodongguaAIGC/X-R1-3B' \
#     --dataset_name='HuggingFaceH4/MATH-500' \
# 	--output_name='./output/result_benchmark_math500'  \
# 	--max_output_tokens=1024 \
# 	--num_gpus=1


from datasets import load_dataset, Dataset, DatasetDict
from vllm import LLM, SamplingParams
import argparse
import json
from utils.prepare_dataset import SYSTEM_PROMPT, SYSTEM_PROMPT_TAG
from rewards import eval_answer_reward, eval_thinking_reward
# import torch
import re
from transformers import AutoTokenizer 



def format_reward(completion):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    matches = re.match(pattern, completion)
    rewards = 1.0 if matches else 0.0 
    return rewards


def create_dataset(dataset_name, tokenizer, tag):
    dataset = load_dataset(dataset_name, split='test')

    def make_conversation(example):
        if tag:
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT_TAG},
                    {"role": "user", "content": example["problem"]},
                ],
            }
        else:
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["problem"]},
                ]
            }

    dataset = dataset.map(make_conversation)

    def make_latex(example):
        example["answer"] = '$' + str(example["answer"]) + '$'
        return example

    dataset = dataset.map(make_latex)

    def format_function(example):
        example['prompt'] = tokenizer.apply_chat_template(example['prompt'], tokenize = False, add_generation_prompt = True )
        return example
    
    dataset = dataset.map(format_function, batched = False)
        
    return dataset


def vllm_generate(model_name, output_name, dataset_name, num_gpus, max_output_tokens):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # evaluation dataset
    dataset = create_dataset(dataset_name, tokenizer, args.tag)
    print(dataset)


    answers = []
    prompts = []
    processes = []
    count = 0
    for data in dataset:
        answers.append(data['answer'])
        prompts.append(data['prompt'])
        processes.append(data['process']) if 'process' in data else processes.append('')
        count += 1
        if count >= args.num_samples:
            break

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.7,
                                     max_tokens=max_output_tokens,
                                     n = args.num_generation,
                                     )
    # Create LLM object
    llm = LLM(model=model_name,  # replace your own model
                dtype='bfloat16',
                tensor_parallel_size=num_gpus,  # number of gpu
                gpu_memory_utilization=0.7,  # prevent OOM
                trust_remote_code=True,
                # use_cache=False,
              )

    # # vllm generation
    outputs = llm.generate(prompts,
                           sampling_params,)
    pro_scores = []
    acc_scores = []
    format_scores = []
    result_all = []
    total_acc = 0
    total_format = 0
    for output, gold_answer, gold_process in zip (outputs, answers, processes):
        prompt = output.prompt
        print("Prompt: ", prompt)

        for output_completion in output.outputs:
            completion = output_completion.text
            print("completion:", completion)
            if args.reward_function == 'eval_answer_reward':
                acc_score, _ = eval_answer_reward(completion, gold_answer, args.tag, silence = False)
                pro_score = 0
            elif args.reward_function == 'eval_answer_thinking_reward':
                acc_score, _ = eval_answer_reward(completion, gold_answer, args.tag, silence = False)
                pro_score, step_end_pos = eval_thinking_reward(completion, gold_answer, gold_process, args.tag, cv_type = args.cv_type, silence = True)
            acc_scores.append(acc_score)
            pro_scores.append(pro_score)
            total_acc = total_acc + acc_score

            format_score = format_reward(completion)
            format_scores.append(format_score)
            total_format = total_format + format_score

            # print('format score', format_score)
            # print('accuracy score', acc_score)
            # print('-'*100)

            result_all.append({
                'prompt': prompt, 
                'completion': completion, 
                'gold answer': gold_answer, 
                'acc scores': acc_score,  
                'pro scores': pro_score,
                'format score': format_score, 
            })

    print('='*100)
    print('eval num: ', len(acc_scores))
    print('eval acc: ', total_acc / len(acc_scores))
    print('eval pro: ', sum(pro_scores) / len(pro_scores))
    print('eval format: ',total_format / len(format_scores))

    # Calculate the average pro scores for those completions with acc score == 0
    pro_scores_zero_acc = [pro_score for pro_score, acc_score in zip(pro_scores, acc_scores) if acc_score == 0]
    # Calculate the average pro scores for those completions with acc score == 1
    pro_scores_one_acc = [pro_score for pro_score, acc_score in zip(pro_scores, acc_scores) if acc_score == 1]
    print('eval pro for acc == 0: ', sum(pro_scores_zero_acc) / len(pro_scores_zero_acc) if pro_scores_zero_acc else 0)
    print('eval pro for acc == 1: ', sum(pro_scores_one_acc) / len(pro_scores_one_acc) if pro_scores_one_acc else 0)

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
    parser.add_argument('--tag', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='tag or not')
    parser.add_argument('--cv_type', type=str, default="num",
                        help='which type of thinking reward to use, e.g., num, regex, gsc')
    parser.add_argument('--num_generation', type=int, default=1,
                        help='number of responses generated for each prompt')
    parser.add_argument('--num_samples', type=int, default=99999999,
                        help='number of samples to evaluate')
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