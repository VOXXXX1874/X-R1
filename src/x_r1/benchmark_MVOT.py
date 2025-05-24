from datasets import load_dataset
from vllm import LLM, SamplingParams
import argparse
import json
from utils.majority_voting import intermediate_majority_voting
from rewards import eval_answer_reward_MVOT
# import torch
import re
from transformers import AutoTokenizer 

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


def vllm_generate(model_name, output_name, dataset_name, num_gpus, max_output_tokens):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # evaluation dataset
    dataset = create_dataset(dataset_name, tokenizer)
    print(dataset)


    answers = []
    prompts = []
    processes = []
    for data in dataset:
        answers.append(data['answer'])
        prompts.append(data['prompt'])
        processes.append(data['process']) if 'process' in data else processes.append('')
        #break

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.7,
                                     max_tokens=max_output_tokens,
                                     stop="</answer>",
                                     )
    # Create LLM object
    llm = LLM(model=model_name,  # replace your own model
                dtype='bfloat16',
                tensor_parallel_size=num_gpus,  # number of gpu
                gpu_memory_utilization=0.7,  # prevent OOM
                trust_remote_code=True,
                # use_cache=False,
              )

    pro_scores = []
    acc_scores = []
    result_all = []
    total_acc = 0
    total_format = 0
    for prompt, gold_answer, gold_process in zip (prompts, answers, processes):
        selected_context = prompt
        for _ in range(args.max_steps):
            # If there is <final> tag in the context, we stop the generation
            if selected_context[-1]["content"].find("</final>") != -1:
                break
            # Tokenize the selected context
            print('selected context: ', selected_context)
            input_ids = tokenizer.apply_chat_template(selected_context, tokenize = False, add_generation_prompt = True )
            # vllm generation
            outputs = llm.generate([input_ids] * args.num_generation,
                                sampling_params, use_tqdm=False)
            # collect all the intermediate results inside <answer>...</answer>
            intermediate_results = []
            # go through all the outputs
            for output in outputs:
                completion = output.outputs[0].text
                # search for the <answer> tag
                answer = re.search(r"</think>(.*)", completion, re.DOTALL)
                if answer:
                    # Append the answer to the list
                    intermediate_results.append(answer.group(1))

            if len(intermediate_results) == 0:
                break
            # Perform majority voting on the intermediate results
            # <final> is also part of majority voting
            if selected_context[-1]['role'] == 'user':
                selected_context.append({"role": "assistant", "content": outputs[intermediate_majority_voting(intermediate_results)].outputs[0].text + "</answer>"})
            elif selected_context[-1]['role'] == 'assistant':
                selected_context[-1]["content"] += outputs[intermediate_majority_voting(intermediate_results)].outputs[0].text + "</answer>"

        completion = selected_context[-1]["content"]
        # if the max_step = 0
        if args.max_steps < 1:
            input_ids = tokenizer.apply_chat_template(selected_context, tokenize = False, add_generation_prompt = True )
            final_output = llm.generate([input_ids],
                                        SamplingParams(temperature=0.7,
                                            max_tokens=max_output_tokens,
                                            ),
                                        use_tqdm=False,)
            completion = final_output[0].outputs[0].text
        # if the final answer is not in the context, inference the final answer
        elif selected_context[-1]["content"].find("</final>") == -1:
            input_ids = tokenizer.apply_chat_template(selected_context, tokenize = False, add_generation_prompt = True )
            final_output = llm.generate([input_ids],
                                        SamplingParams(temperature=0.7,
                                            max_tokens=max_output_tokens,
                                            ),
                                        use_tqdm=False,)
            completion += final_output[0].outputs[0].text
        
        if args.reward_function == 'eval_answer_reward':
            acc_score, _ = eval_answer_reward_MVOT(completion, gold_answer, silence = False)
            pro_score = 0
        elif args.reward_function == 'eval_answer_thinking_reward':
            acc_score, _ = eval_answer_reward_MVOT(completion, gold_answer, silence = False)
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