# Import libraries to read json files and manipulate data
import json
# regular expression library for pattern matching
import re
import random

# Read the json file train.json
with open("MATH-500/raw/test.json", "r") as f:
    math_qa_dataset = json.load(f)

# Initialize an empty list to store the processed data
num_math_qa_dataset = []
# Regular expression to match numbers
number_pattern = re.compile(r'\d+')
# distribution of the appearance of numbers
num_distribution = {}
# numbers in problem and solution
num_in_ps = []
# Iterate through each item in the dataset
for item in math_qa_dataset:
    # Extract the question, solution, and answer from the item
    question = item['problem']
    solution = item['solution']
    answer = item['answer']
    # Find all numbers in the question using the regular expression
    numbers_question = number_pattern.findall(question)
    # Convert the numbers to integers
    numbers_question = [int(num) for num in numbers_question]
    # Convert them into set
    numbers_question = set(numbers_question)
    # Find all numbers in the solution using the regular expression
    numbers_solution = number_pattern.findall(solution)
    # Convert the numbers to integers
    numbers_solution = [int(num) for num in numbers_solution]
    # Convert them into set
    numbers_solution = set(numbers_solution)
    # Find all numbers in the answer using the regular expression
    numbers_answer = number_pattern.findall(answer)
    # Convert the numbers to integers
    numbers_answer = [int(num) for num in numbers_answer]
    # Convert them into set
    numbers_answer = set(numbers_answer)

    # Find the numbers that are in the solution but not in the question and the answer
    numbers_solution_not_in_qa = numbers_solution - numbers_question - numbers_answer
    # If there is no number in "numbers_solution_not_in_qa"
    if not numbers_solution_not_in_qa:
        target_cv = numbers_solution - numbers_answer
    else:
        target_cv = numbers_solution_not_in_qa
    num_in_ps.append(numbers_question.union(numbers_solution))

    # Record the distribution of the appearance of numbers
    for num in target_cv:
        if num not in num_distribution:
            num_distribution[num] = 0
        num_distribution[num] += 1

    # Add the numbers into the item as "num 1 <sep> num 2 <sep> ..."
    item['process'] = ' <target> ' + ' <sep> '.join([str(num) for num in target_cv])
    # Append the item to the processed dataset
    num_math_qa_dataset.append(item)

# Iterate through each item in the processed dataset
for item, ps_record in zip(num_math_qa_dataset, num_in_ps):
    # Sample 10 numbers according to the num_distribution as weight
    sampled_numbers = random.choices(
        list(num_distribution.keys()), 
        weights=list(num_distribution.values()), 
        k=10
    )
    # add 0 1 2 3 4 5 6 to the sampled numbers
    sampled_numbers += [0, 1, 2, 3, 4, 5, 6]
    # Convert the sampled numbers to set to remove duplicates
    sampled_numbers = set(sampled_numbers)
    # Convert the sampled numbers to list and sort them
    sampled_numbers = sorted(sampled_numbers)
    # Remove numbers that are already in the target_cv
    sampled_numbers = [num for num in sampled_numbers if num not in ps_record]
    # Add the sampled numbers to the item as "num 1 <sep> num 2 <sep> ..."
    item['process'] += ' <trap> ' + ' <sep> '.join([str(num) for num in sampled_numbers])


# Save the processed dataset to a new json file
with open("num_math_qa_dataset.json", "w") as f:
    json.dump(num_math_qa_dataset, f, indent=4)
    