from transformers import AutoModelForCausalLM, AutoTokenizer

#model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_name = "output/Qwen2.5-1.5B-MVOT"
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
    {"role": "user", "content": question}
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
    temperature=0.7,
    do_sample=True,
    stop_strings=["</answer>"],
    tokenizer=tokenizer,
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)