from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
# import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import json

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

    def format_function(example):
        example['prompt'] = tokenizer.apply_chat_template(example['prompt'], tokenize = False, add_generation_prompt = True )
        return example
    
    dataset = dataset.map(format_function, batched = False)
        
    return dataset

model_name = "records/Qwen2.5-1.5B-MVOT"

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

dataset = create_dataset('src/cv_extraction/MATH-500/exp', tokenizer)

prompts = []
for data in dataset:
    prompts.append(data['prompt'])

# vllm generation
outputs = llm.generate( prompts, 
                        sampling_params=sampling_params,
                        )

results = []

for output, data in zip(outputs, dataset):
    results.append({
        "problem": data['problem'],
        "answer": data['answer'],
        "solution": data['solution'],
        "output": [output.outputs[i].text for i in range(len(output.outputs))],
    })

# Save the result to voting.json
with open('voting.json', 'w') as f:
    json.dump(results, f, indent=4)