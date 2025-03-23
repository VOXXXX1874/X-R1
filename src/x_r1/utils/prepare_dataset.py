from datasets import load_dataset

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

# Format into conversation
def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }
    
def make_latex(example):
    example["solution"] = '$' + str(example["solution"]) + '$'
    return example

def prepare_dataset(dataset_name, split):
    dataset = load_dataset(dataset_name)

    # align the dataset
    if dataset_name == "FreedomIntelligence/medical-o1-verifiable-problem":
        dataset = dataset.rename_columns({
            "Open-ended Verifiable Question": "problem",
            "Ground-True Answer": "solution"
        })

    # if DeepScaleR-Preview-Dataset in the name of the dataset, then we need to remove the solution column and rename answer column to solution
    if "DeepScaleR-Preview-Dataset" in dataset_name:
        # Take half of the dataset
        dataset[split] = dataset[split].select(range(len(dataset[split])//2))

        dataset = dataset.remove_columns("solution")
        dataset = dataset.rename_columns({"answer": "solution"})
        
        dataset = dataset.map(make_latex)
    elif dataset_name == "src/fol_r1/gsc":
        dataset = dataset.rename_columns({"answer": "solution"})
        dataset = dataset.map(make_latex)

    dataset = dataset.map(make_conversation)
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    return dataset

def prepare_quick_eval_dataset(dataset_name):
    quick_eval_dataset = load_dataset(dataset_name)

    if dataset_name == "Maxwell-Jia/AIME_2024":
        quick_eval_dataset = quick_eval_dataset.remove_columns("Solution")
        quick_eval_dataset = quick_eval_dataset.remove_columns("ID")
        quick_eval_dataset = quick_eval_dataset.rename_column("Answer", "solution")
        quick_eval_dataset = quick_eval_dataset.rename_column("Problem", "problem")
        quick_eval_dataset = quick_eval_dataset.map(make_latex)
        quick_eval_dataset = quick_eval_dataset.map(make_conversation)
        quick_eval_dataset = quick_eval_dataset['train']
        # display one example from the dataset
        print('Example from quick_eval_dataset:', quick_eval_dataset[0])
    elif dataset_name == "HuggingFaceH4/MATH-500":
        quick_eval_dataset = quick_eval_dataset.remove_columns("solution")
        quick_eval_dataset = quick_eval_dataset.rename_column("answer", "solution")
        quick_eval_dataset = quick_eval_dataset.map(make_latex)
        quick_eval_dataset = quick_eval_dataset.map(make_conversation)
        quick_eval_dataset = quick_eval_dataset['test']
        # display one example from the dataset
        print('Example from quick_eval_dataset:', quick_eval_dataset[0])
    else:
        raise ValueError(f"Invalid quick eval dataset: {dataset_name}")
    
    return quick_eval_dataset