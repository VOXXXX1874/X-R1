# Import libraries to read json files and manipulate data
import json
# regular expression library for pattern matching
import re
import random

def extract_numbers_from_string(s):
    """
    Extracts all numbers from a given string and returns them as a list of integers.
    """
    # Regular expression to match numbers
    number_pattern = re.compile(r'\d+')
    # Find all numbers in the string
    numbers = number_pattern.findall(s)
    # Convert the numbers to integers
    return [int(num) for num in numbers]

# Read the json file train.json
with open("../XR1-7500/extend/train.json", "r") as f:
    math_qa_dataset = json.load(f)

# Initialize an empty list to store the processed data
num_math_qa_dataset = []
# distribution of the appearance of numbers
num_distribution = {}
# numbers in problem and solution
num_in_ps = []
# Iterate through each item in the dataset
for item in math_qa_dataset:
    if len(item['correct_responses']) / (len(item['wrong_responses']) + len(item['correct_responses'])) > 0.4 or len(item['correct_responses']) / (len(item['wrong_responses']) + len(item['correct_responses'])) < 0.2:
        continue
    # Extract the question, solution, and answer from the item
    question = item['problem']
    solution = item['solution']
    correct_response = item.pop('correct_responses', [])
    # Remove the content after "\box" in the solution
    solution = solution.split('\\box')[0].strip()
    correct_response = [resp.split('\\box')[0].strip() for resp in correct_response]
    # Find all numbers in the question
    numbers_question = set(extract_numbers_from_string(question))
    # Find all numbers in the solution using the regular expression
    numbers_solution = set(extract_numbers_from_string(solution))
    # Find all numbers in the correct response
    for resp in correct_response:
        numbers_solution.update(extract_numbers_from_string(resp))
    
    # Find the numbers that are in the solution but not in the question and the answer
    target_cv = numbers_solution - numbers_question
    if len(target_cv) < 5:
        continue
    num_in_ps.append(" <sep> ".join([str(num) for num in list(numbers_solution.union(numbers_question))]))

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
    wrong_response = item.pop('wrong_responses', [])
    wrong_response = [resp.split('\\box')[0].strip() for resp in wrong_response]
    # Find all numbers in the wrong response
    numbers_wrong_response = set()
    for resp in wrong_response:
        numbers_wrong_response.update(extract_numbers_from_string(resp))
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
    # merge the sampled numbers with the numbers in the wrong response
    sampled_numbers.update(numbers_wrong_response)
    # Convert the sampled numbers to list and sort them
    sampled_numbers = sorted(sampled_numbers)
    # Remove numbers that are already in the target_cv
    sampled_numbers = [num for num in sampled_numbers if str(num) not in ps_record]
    # Add the sampled numbers to the item as "num 1 <sep> num 2 <sep> ..."
    item['process'] += ' <trap> ' + ' <sep> '.join([str(num) for num in sampled_numbers])


# Save the processed dataset to a new json file
with open("num_math_qa_dataset.json", "w") as f:
    json.dump(num_math_qa_dataset, f, indent=4)
    