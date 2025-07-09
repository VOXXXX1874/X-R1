from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
# import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import json
from math_verify import LatexExtractionConfig, parse, verify
from sympy import nan, zoo

def outcome_reward(answer, solution):
    gold_parsed = parse(
        solution,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
    answer_parsed = parse(
        answer,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
    if len(answer_parsed) == 0:
        answer_parsed = parse(
            '$' + answer +'$',
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
    if len(answer_parsed) != 0 and (answer_parsed[0] == nan or answer_parsed[0] == zoo):
        return gold_parsed, 'nan', 0.0

    reward = float(verify(answer_parsed, gold_parsed))

    return gold_parsed, answer_parsed, reward

def create_dataset(dataset_name, tokenizer):
    dataset = load_dataset(dataset_name, split='train')

    def make_conversation(example):
        return {
            "prompt": [
                {"role": "user", "content": example["problem"]},
            ],
        }

    dataset = dataset.map(make_conversation)

    def format_function(example):
        example['prompt'] = tokenizer.apply_chat_template(example['prompt'], tokenize = False, add_generation_prompt = True )
        return example
    
    dataset = dataset.map(format_function, batched = False)
        
    return dataset

model_name = "Qwen/Qwen2.5-1.5B-instruct"

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.7,
                                    max_tokens=4096,
                                    n = 10
                                    )
# Create LLM object
llm = LLM(model=model_name,  # replace your own model
            dtype='bfloat16',
            tensor_parallel_size=1,  # number of gpu
            gpu_memory_utilization=0.7,  # prevent OOM
            trust_remote_code=True,
            )

tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = create_dataset('src/cv_extraction/XR1-750/raw', tokenizer)

prompts = []
count = 0
for data in dataset:
    prompts.append(data['prompt'])
    count += 1
    #if count == 50:
    #    break

# vllm generation
outputs = llm.generate( prompts, 
                        sampling_params=sampling_params,
                        )

results = []

for output, data in zip(outputs, dataset):
    correct_responses = []
    wrong_responses = []
    for i in range(len(output.outputs)):
        if outcome_reward(output.outputs[i].text, data['solution'])[2] > 0.5:
            correct_responses.append(output.outputs[i].text)
        else:
            wrong_responses.append(output.outputs[i].text)

    results.append({
        'problem': data['problem'],
        'solution': data['solution'],
        'level': data['level'],
        'type': data['type'],
        'correct_responses': correct_responses,
        'wrong_responses': wrong_responses,
    })

# Save the result to voting.json
with open('extended.json', 'w') as f:
    json.dump(results, f, indent=4)