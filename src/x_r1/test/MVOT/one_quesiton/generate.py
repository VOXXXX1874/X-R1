from transformers import AutoModelForCausalLM, AutoTokenizer

#model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_name = "Qwen/Qwen2.5-3B-Instruct"
device = "cuda" # the device to load the model onto

# Read question from question.txt
with open("src/x_r1/test/MVOT/one_quesiton/question.md", "r") as file:
    question = file.read()

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# CoT
messages = [
    {"role": "system", "content": "You are a helpful AI Assistant that provides well-reasoned and detailed responses."},
    {"role": "user", "content": "There is a math question:\n" + question + "\n You are not required to get the final answer. Please provide a detailed reasoning process and stop at the most convincing step."}
]

print(messages)

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=16384,
    temperature=0.3,
    do_sample=True
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)