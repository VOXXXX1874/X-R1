from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
# import torch
from transformers import AutoTokenizer 

#model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
device = "cuda" # the device to load the model onto

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
            # use_cache=False,
            )

tokenizer = AutoTokenizer.from_pretrained(model_name)

system_prompt = (
    "You are a helpful AI Assistant."
    "You will be given a math question, but you are not required to get the final answer."
    "Please provide a detailed reasoning process and stop at the most convincing step with an intermediate answer."
    "The Assistant encloses its reasoning in <think> </think>, the answer in <answer> </answer>."
    "There is an example:\n"
    "**Question**\n"
    "What are all values of $p$ such that for every $q>0$, we have $$\\frac{3(pq^2+p^2q+3q^2+3pq)}{p+q}>2p^2q?$$ Express your answer in interval notation in decimal form.\n"
    "**Answer**\n"
    "*Step 1*\n"
    "<think> First we'll simplify that complicated expression. We attempt to factor the numerator of the left side: \\begin{align*} pq^2+p^2q+3q^2+3pq &= q(pq + p^2 + 3q + 3p) \\\\ &= q[ p(q+p) + 3(q+p) ] \\\\ &= q(p+3)(q+p). \\end{align*} </think>\n"
    "<answer> Numerator of the left side: $q(p+3)(q+p)$ </answer>\n"
    "*Step 2*\n"
    "<think> Substituting this in for the numerator in our inequality gives $$\\frac{3q(p+3)(p+q)}{p+q}>2p^2q.$$We note that left hand side has $p+q$ in both the numerator and denominator. We can only cancel these terms if $p+q \\neq 0.$ Since we're looking for values of $p$ such that the inequality is true for all $q > 0,$ we need $p \\geq 0$ so that $p + q \\neq 0.$\n"
    "<answer> We can derive $p \\geq 0$ </answer>\n"
    "*Step 3*\n"
    "<think> Also because this must be true for every $q>0$, we can cancel the $q$'s on both sides. This gives \\begin{align*} 3(p+3)&>2p^2\\Rightarrow\\\\ 3p+9&>2p^2 \\Rightarrow\\\\ 0&>2p^2-3p-9. \\end{align*} </think>\n"
    "<answer> The inequality can be simplified to $2p^2-3p-9<0$ </answer>\n"
    "*Step 4*\n"
    "<think> Now we must solve this quadratic inequality. We can factor the quadratic as $2p^2-3p-9=(2p+3)(p-3)$. The roots are $p=3$ and $p=-1.5$. Since a graph of this parabola would open upwards, we know that the value of $2p^2 - 3p - 9$ is negative between the roots, so the solution to our inequality is $-1.5<p<3.$ But we still need $0 \\leq p,$ so in interval notation the answer is $\\boxed{[0,3)}$.</think>\n"
    "<answer> The final answer is $\\boxed{[0,3)}$ </answer>\n"
)

# CoT
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": question}
]

print(messages)

prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True )

# vllm generation
outputs = llm.generate([prompt],
                        sampling_params=sampling_params,
                        device=device,
                        gpu_memory_utilization=1.0
                        )

# Print the output
print(outputs[0].text)