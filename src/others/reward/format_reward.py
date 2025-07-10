import re

def format_reward(completions, silence=False, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]

    rewards = [1.0 if match else 0.0 for match in matches]
    if not silence:
        print('-'*100)
        print('\nformat rewards:', rewards)
    return rewards

# Read solution from solution.txt
with open("src/x_r1/test/reward/solution.md", "r") as file:
    solution = file.read()

# Test the reward function
completions = [solution]
rewards = format_reward(completions)
print(rewards)