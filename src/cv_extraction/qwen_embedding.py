# Requires vllm>=0.8.5
import torch
import vllm
from vllm import LLM

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

# Each query must come with a one-sentence instruction that describes the task
task = 'Given a math expression, retrieve another expression that is semantically similar to it.'

queries = [
    get_detailed_instruct(task, '\n\n\\[\n\\frac{a}{25-a} + \\frac{b}{65-b} + \\frac{c}{60-c} = 7\n\\]\n\n'),
    get_detailed_instruct(task, '\n\n\\[\n\\frac{5}{25-a} + \\frac{13}{65-b} + \\frac{12}{60-c}\n\\]\n\n')
]
# No need to add instruction for retrieval documents
documents = [
    "\\[\n\\frac{x}{25-x} + \\frac{y}{65-y} + \\frac{z}{60-z} = 7\n\\]",
    "\\(3x(x+1) + 7(x+1)\\)"
]
input_texts = queries + documents

model = LLM(model="Qwen/Qwen3-Embedding-8B", task="embed")

outputs = model.embed(input_texts)
embeddings = torch.tensor([o.outputs.embedding for o in outputs])
scores = (embeddings[:2] @ embeddings[2:].T)
print(scores.tolist())
# [[0.7620252966880798, 0.14078938961029053], [0.1358368694782257, 0.6013815999031067]]