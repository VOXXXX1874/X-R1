from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
# import torch
from transformers import AutoTokenizer 
import re

#model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_name = "records/Qwen2.5-1.5B-MVOT"

# Read question from question.txt
with open("src/x_r1/test/MVOT/one_quesiton/question.md", "r") as file:
    question = file.read()

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.7,
                                    max_tokens=4096,
                                    stop="</answer>",
                                    )
# Create LLM object
llm = LLM(model=model_name,  # replace your own model
            dtype='bfloat16',
            tensor_parallel_size=1,  # number of gpu
            gpu_memory_utilization=0.7,  # prevent OOM
            trust_remote_code=True,
            )

tokenizer = AutoTokenizer.from_pretrained(model_name)

# CoT
messages = [
    {"role": "user", "content": question}
]

print(messages)

prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True )

# vllm generation
outputs = llm.generate([prompt]* 500,  # repeat the prompt for 100 times
                        sampling_params=sampling_params,
                        )

answer_list = []

for output in outputs:
    # parse <answer>
    completion = output.outputs[0].text
    answer = re.search(r"<answer>(.*)", completion, re.DOTALL)
    if answer:
        answer = answer.group(1)
    else:
        answer = "No answer found"
    answer_list.append(answer)

# Save the answer to voting.txt
with open("voting.txt", "w") as file:
    for answer in answer_list:
        file.write("<answer>" + answer + "</answer>\n")