import json
import re
from transformers import AutoTokenizer
import random

# Read the "result.json" file
with open("records/dataset3_DS.json", "r") as f:
    qa_dataset = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

system_prompt = ("A conversation between User and Assistant. The user asks a question, and the Assistant solves it."
    "The assistant first thinks about the reasoning process in the mind and then records the intermediate results."
    "The recorded intermediate results should be enough for the reasoning process in the later steps."
    "The reasoning process, record, and final answer are enclosed within"
    "<think> </think>, <record> </record>, and <answer> </answer> tags, respectively.")

# for each sample in train.json, process the segment
segmented_dataset = []
for i, qa in enumerate(qa_dataset):
    # get the answer
    answer = qa["messages"][-1]["content"]
    # Find all the positions of "<record>" in the answer
    record_positions = [m.start() for m in re.finditer(r"<record>", answer)]
    # Find all the positions of "</record>" in the answer
    end_record_positions = [m.start() for m in re.finditer(r"</record>", answer)]
    # Find all the positions of "<think>" in the answer
    think_positions = [m.start() for m in re.finditer(r"<think>", answer)]
    # Find all the positions of "</think>" in the answer
    end_think_positions = [m.start() for m in re.finditer(r"</think>", answer)]

    if len(record_positions) != len(end_record_positions) or len(think_positions) != len(end_think_positions) or len(record_positions) != len(think_positions):
        print("Error: The number of <record> and </record> or <think> and </think> tags do not match.")
        print(f"Skipping sample {i} due to mismatched tags.")
        continue

    # For each record, extract the segment and remove all the content in <think> and </think>
    for i in range(len(record_positions)):
        new_qa = {}
        new_answer = ""
        for j in range(0, i):
            if j == 0:
                new_answer += answer[0:think_positions[j]] + answer[end_think_positions[j] + len("</think>"): end_record_positions[j]  + len("</record>")]
            else:
                new_answer += answer[end_record_positions[j-1] + len("</record>"): think_positions[j]] + answer[end_think_positions[j] + len("</think>"): end_record_positions[j] + len("</record>")]
        if i == 0:
            new_answer += answer[0:end_record_positions[i] + len("</record>")]
        elif i == len(record_positions) - 1:
            new_answer += answer[end_record_positions[i-1] + len("</record>"):]
        else:
            new_answer += answer[end_record_positions[i-1] + len("</record>"): end_record_positions[i] + len("</record>")]
        new_answer = new_answer.strip().replace("\n\n", "\n")
        if new_answer:
            # construct the new qa entry
            message = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": qa["messages"][0]["content"]},
                {"role": "assistant", "content": new_answer}
            ]
            # apply chat template to the new message
            text = tokenizer.apply_chat_template(
                message, tokenize=False, continue_final_message=True
            )
            new_qa["text"] = text
            if i!= 0:
                # Tokenize the text and get offset mapping
                inputs = tokenizer(text, return_tensors="pt", padding=True, return_offsets_mapping=True)
                offsets = inputs.pop("offset_mapping")[0]
                # Find the second last "</record>" tag in the text
                second_last_record_end = text[:text.rfind("</record>")].rfind("</record>") + len("</record>") + 1
                # if the tokens is in the range of the first "<record>" and second last "</record>", set the labels to 0, otherwise 1
                objective_mask = []
                for token_idx, (start, end) in enumerate(offsets):
                    if end <= second_last_record_end:
                        objective_mask.append(0)
                    else:
                        objective_mask.append(1)
            else:
                objective_mask = [1] * len(tokenizer(text, return_tensors="pt").input_ids[0])

            new_qa["objective_mask"] = objective_mask
            segmented_dataset.append(new_qa)
    
    #break

# shuffle the segmented dataset
random.shuffle(segmented_dataset)

# Save the segmented dataset to a new file
with open("segmented_dataset.json", "w") as f:
    json.dump(segmented_dataset, f)