"""Reward functions for GRPO training."""

import re
from math_verify import LatexExtractionConfig, parse, verify
from math_verify.parser import *
from sympy import nan, zoo
from utils.regex_math import critical_value_reward_regex
from utils.gsc import critical_value_reward_gsc
from utils.num import critical_value_reward_num

def extract_final(completion):
    """Extract the final answer from the completion."""
    # Use regex to find the answer in the format <answer>...</answer>
    pattern = r"<final>(.*?)</final>"
    match = re.search(pattern, completion, re.DOTALL)
    if match:
        answer = match.group(1)
        return answer
    else:
        return "No answer found"

def extract_answer(text):
    """Extract content between <answer> tags."""
    if text is None:
        return ""
    match = re.search(r'<answer>(.*?)</answer>$', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def extract_thinking(text):
    """Extract content between <think> tags."""
    if text is None:
        return ""
    match = re.search(r'^<think>(.*?)</think>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def outcome_reward(answer, solution):
    gold_parsed = parse(
        solution,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
    answer_parsed = parse(
        answer,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
    if len(answer_parsed) == 0:
        answer_parsed = parse(
            '$' + answer +'$',
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
    if len(answer_parsed) != 0 and (answer_parsed[0] == nan or answer_parsed[0] == zoo):
        return gold_parsed, 'nan', 0.0

    reward = float(verify(answer_parsed, gold_parsed))

    return gold_parsed, answer_parsed, reward

# for training
def accuracy_reward(completions, solution, tag=True, silence=False, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        # First try latex parsing
        answer = extract_answer(content) if tag else content
        gold_parsed, answer_parsed, reward = outcome_reward(answer, sol)
        # print('\nprompt:', prompt)
        if not silence:
            print('-'*100)
            try:
                print('\nanswer_parsed:', answer_parsed, '\ngold_parsed:', gold_parsed, '\nreward:', reward)
            except:
                print('\nanswer_parsed:', 'NaN', '\ngold_parsed:', gold_parsed, '\nreward:', reward)
        rewards.append(reward)

    if not silence:
        print('\naccuracy rewards:', rewards)

    return rewards, []

def thinking_reward(completions, solution, process, tag=True, silence=False, cv_type = "num", **kwargs):
    """Reward function that checks if the completion is the same as the ground truth and assign partial reward for crucial thinking results."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    steps_end_pos = []
    for content, pro in zip(contents, process):
        # parse ground truth process
        thinking_completion = extract_thinking(content) if tag else content
        if cv_type == "gsc":
            reward, step_end_pos = critical_value_reward_gsc(thinking_completion, pro)
        elif cv_type == "regex":
            reward, step_end_pos = critical_value_reward_regex(thinking_completion, pro)
        elif cv_type == "num":
            reward, step_end_pos = critical_value_reward_num(thinking_completion, pro)
        else:
            raise ValueError(f"Unknown type: {cv_type}")
        rewards.append(reward)
        steps_end_pos.append(step_end_pos)
    if not silence:
        print('\nThinking rewards:', rewards, '\nCorrect thinking steps end pos:', steps_end_pos)

    return rewards, steps_end_pos
        
def accuracy_thinking_reward(completions, solution, process, tag=True, silence=False, cv_type="num", **kwargs):
    """Reward function that checks if the completion is the same as the ground truth and assign partial reward for crucial thinking results."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    steps_end_pos = []
    for content, sol, pro in zip(contents, solution, process):
        # outcome reward
        answer = extract_answer(content) if tag else content
        gold_parsed, answer_parsed, reward = outcome_reward(answer, sol)
        # process reward
        if reward == 0.0:
            # parse ground truth process
            thinking_completion = extract_thinking(content)  if tag else content
            if cv_type == "gsc":
                step_end_pos = critical_value_reward_gsc(thinking_completion, pro)
            elif cv_type == "regex":
                step_end_pos = critical_value_reward_regex(thinking_completion, pro)
            elif cv_type == "num":
                step_end_pos = critical_value_reward_num(thinking_completion, pro)
            else:
                raise ValueError(f"Unknown type: {cv_type}")
            reward = reward * 0.6
            steps_end_pos.append(step_end_pos)
        else:
            steps_end_pos.append(0)

        rewards.append(reward)
        
    if not silence:
        print('\naccuracy thinking rewards:', rewards, '\nCorrect thinking steps end pos:', steps_end_pos)

    return rewards, steps_end_pos

# for benchmark.py
def eval_answer_reward(completion, solution, tag=False, silence=False, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    # outcome reward
    completion = extract_answer(completion) if tag else completion
    gold_parsed, answer_parsed, reward = outcome_reward(completion, solution)
    if not silence:
        print('-'*100)
        try:
            print('\nanswer_parsed:', answer_parsed, '\ngold_parsed:', gold_parsed, '\nreward:', reward)
        except:
            print('\nanswer_parsed:', 'NaN', '\ngold_parsed:', gold_parsed, '\nreward:', reward)

    return reward, 0


# for benchmark_MVOT.py
def eval_answer_reward_MVOT(completion, solution, silence=False, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    # outcome reward
    completion = extract_final(completion)
    gold_parsed, answer_parsed, reward = outcome_reward(completion, solution)
    if not silence:
        print('-'*100)
        try:
            print('\nanswer_parsed:', answer_parsed, '\ngold_parsed:', gold_parsed, '\nreward:', reward)
        except:
            print('\nanswer_parsed:', 'NaN', '\ngold_parsed:', gold_parsed, '\nreward:', reward)

    return reward, 0

def eval_thinking_reward(completion, solution, process, tag=False, cv_type = "num", silence=False, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth and assign partial reward for crucial thinking results."""
    thinking_completion = extract_thinking(completion) if tag else completion
    if cv_type == "gsc":
        reward, step_end_pos = critical_value_reward_gsc(thinking_completion, process)
    elif cv_type == "regex":
        reward, step_end_pos = critical_value_reward_regex(thinking_completion, process)
    elif cv_type == "num":
        reward, step_end_pos = critical_value_reward_num(thinking_completion, process)
    else:
        raise ValueError(f"Unknown type: {cv_type}")
    if not silence:
        print('\nthinking_reward:', reward, '\nstep_end_pos:', step_end_pos)
    return reward, step_end_pos


def format_reward(completions, silence=False, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]

    rewards = [1.0 if match else 0.0 for match in matches]
    if not silence:
        print('\nformat rewards:', rewards)
        print('-'*100)
    return rewards, []