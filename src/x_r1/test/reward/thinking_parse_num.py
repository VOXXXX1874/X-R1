import re
from math_verify.parser import *
import time

def extract_thinking(text):
    """Extract content between <think> tags."""
    if text is None:
        return ""
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

# Read solution from solution.txt
with open("src/x_r1/test/reward/solution.md", "r") as file:
    solution = file.read()

# Read solution from solution.txt
with open("src/x_r1/test/reward/solution_pm.md", "r") as file:
    solution_pm = file.read()

# Count the time taken to parse and verify the solution
start_time = time.time()

solution_thinking = extract_thinking(solution)
gold_parsed = extract_thinking(solution_pm).split(" <sep> ")

atomic_reward = 1.0/len(gold_parsed)
reward = 0.0
final_end_pos = 0
for gold_parsed_results in gold_parsed:
    if gold_parsed_results.strip() in solution_thinking:
        reward += atomic_reward
        tmp = solution_thinking.index(gold_parsed_results.strip()) + len(gold_parsed_results.strip())
        if tmp > final_end_pos:
            final_end_pos = tmp

print("--- %s seconds ---" % (time.time() - start_time))

print("reward", reward)
print("atomic_reward", atomic_reward)
print("final_end_pos", final_end_pos)
