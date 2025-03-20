from transformers import AutoModelForCausalLM, AutoTokenizer

# Read question from question.txt
with open("src/x_r1/test/consistency/question1/question.txt", "r") as file:
    question = file.read()

# Read output1 from output1.txt
with open("src/x_r1/test/consistency/question1/output1.txt", "r") as file:
    output1 = file.read()

# Read output2 from output2.txt
with open("src/x_r1/test/consistency/question1/output2.txt", "r") as file:
    output2 = file.read()

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = [
    {"role": "user", "content": question}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# concatenate the output1 and output2 to the text

text = text + output1 + "\n\nLet's try another methods to verify the correctness.\n\n" + output2 + "\n\nWait, the derived answer is different. I need to further investigate the issue.\n"

print('Input text:', text)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=16384,
    temperature=1.0, 
    do_sample=True,
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print('Output text:', response)