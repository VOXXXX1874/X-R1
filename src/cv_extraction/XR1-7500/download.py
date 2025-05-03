from datasets import load_dataset

# download the dataset
dataset = load_dataset("xiaodongguaAIGC/X-R1-7500")

# Save the training and testing dataset as json files
dataset["train"].to_json("train.json")
dataset["test"].to_json("test.json")

