"""Reward functions for GRPO training."""

import re
from math_verify import LatexExtractionConfig, parse, verify
from math_verify.parser import *
from sympy import nan, zoo

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

def thinking_parse(
    pred: str,
    extraction_config: Sequence[ExtractionTarget] = [
        LatexExtractionConfig(),
        ExprExtractionConfig(),
    ],
):
    """
    Parses all mathematical expressions appearing in a prediction string.
    Extract possible 'critical value' from the expression.
    """
    try:
        target_res = get_extraction_regexes(extraction_config)
        return extract_target_from_pred(pred, target_res)
    except Exception as e:
        print(f"Error during parsing: {e}")
        return []


def extract_target_from_pred(
    pred: str,
    target_res: list[tuple[list[tuple[re.Pattern[str], int]], ExtractionTarget]],
):
    """Extracts targets from a prediction string using regex patterns.
    Returns all sucesffuly extracted match.
    """
    extracted_predictions = []
    predictions_end_pos = []

    # Get all patterns and sort by priority
    all_patterns = [
        (pattern, target_type, priority)
        for target_patterns, target_type in target_res
        for pattern, priority in target_patterns
    ]

    # Group patterns by priority using itertools.groupby
    sorted_patterns = sorted(all_patterns, key=lambda x: x[2])
    grouped_patterns = list((gr, list(val)) for gr, val in groupby(sorted_patterns, key=lambda x: x[2]))
    _, patterns_group = grouped_patterns[-1]
    # Find all matches for each pattern in this priority group
    matches_with_pos = (
        (match, match.start(), match.end(), target_type)
        for pattern, target_type, _ in patterns_group
        for match in pattern.finditer(pred)
    )

    # Try to extract from each match, starting from rightmost
    for match, _, end_position, target_type in matches_with_pos:
        # Find the last '=' in the match
        last_eq = match.group(0).rfind("=")
        # If there is an '=', extract from the right side of the '=' and perform further extraction
        if last_eq != -1:
            # Convert the match to plain string
            match = match.group(0)
            match = match[last_eq + 1:]
            # If match contains \], add \[ to the beginning
            if "\\]" in match:
                match = "\\[" + match
            # If match contains \), add \( to the beginning
            if "\\)" in match:
                match = "\\(" + match
            # Convert the match back to a regex match object
            pred = match
            matches_with_pos = (
                (match, match.start(), match.end(), target_type)
                for pattern, target_type, _ in patterns_group
                for match in pattern.finditer(pred)
            )
            match, _, _, _ = next(matches_with_pos)
            #print(match)
        # Extract the match
        extracted_match, str_fallback = extract_match(match, target_type)

        if extracted_match is not None:
            extracted_predictions.append(extracted_match)
            predictions_end_pos.append(end_position + len('<think>'))

    return extracted_predictions, predictions_end_pos
    
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

def critical_value_reward(thinking_completion, process):
    reward = 0.0
    final_end_pos = 0
    gold_steps, _ = thinking_parse(
            extract_thinking(process),
            extraction_config=[LatexExtractionConfig()],
        )
    if len(gold_steps) != 0:
        steps_parsed, steps_end_pos = thinking_parse(
            thinking_completion,
            extraction_config=[LatexExtractionConfig()],
        )
        atom_reward = 1.0 / len(gold_steps)
        for gold_step in gold_steps:
            for step_parsed, end_pos in zip(steps_parsed, steps_end_pos):
                if step_parsed == nan or step_parsed == zoo:
                    continue
                if verify(step_parsed, gold_step):
                    reward += atom_reward
                    final_end_pos = end_pos if end_pos > final_end_pos else final_end_pos
                    break
    return reward, final_end_pos

def response_cv_parse_regex(text):
    """Parse the response text to decompose the thinking part."""
    # separate the response into lines
    lines = text.split("\n")
    # remove leading and trailing whitespace from each line
    lines = [line.strip() for line in lines]
    # remove lines with too little words or too much words
    lines = [line for line in lines if len(line) > 50 and len(line) < 1000]
    # record the end position of each line in text
    end_pos = []
    for line in lines:
        # find the position of the line in the text
        pos = text.find(line)
        if pos != -1:
            end_pos.append(pos + len(line))
    # make the lines all lower case
    lines = [line.lower() for line in lines]
    # remove whitespace
    lines = [line.replace(" ", "") for line in lines]
    return lines, end_pos

def gt_cv_parse_regex(text):
    """Parse the ground truth text to decompose the thinking part."""
    # separate the response by <sep>
    lines = text.split("<sep>")
    # remove leading and trailing whitespace from each line
    lines = [line.strip() for line in lines]
    # escape special characters
    lines = [re.escape(line) for line in lines]
    # replace ' ' with '.*' 
    lines = [line.replace("\\ ", ".*") for line in lines]
    # make the lines all lower case
    lines = [line.lower() for line in lines]

    return lines, []

def regex_verify(gt, response):
    """Verify if the response matches the ground truth using regex."""
    # compile the regex pattern
    pattern = re.compile(gt)
    # search for the pattern in the response
    match = pattern.search(response)
    return match is not None

def critical_value_reward_regex(thinking_completion, process):
    response_parsed, response_end_pos = response_cv_parse_regex(thinking_completion,)

    gt_parsed, _ = gt_cv_parse_regex(process,)

    atomic_reward = 1.0/len(gt_parsed)
    reward = 0.0
    final_end_pos = 0
    for gt_element in gt_parsed:
        for response_element, end_pos in zip(response_parsed, response_end_pos):
            if regex_verify(gt_element, response_element):
                reward += atomic_reward
                final_end_pos = end_pos
                break

    return reward, final_end_pos

# for training
def accuracy_reward(completions, solution, silence=False, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        # First try latex parsing
        answer = extract_answer(content)
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

def thinking_reward(completions, solution, process, silence=False, regex = False, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth and assign partial reward for crucial thinking results."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    steps_end_pos = []
    for content, pro in zip(contents, process):
        # parse ground truth process
        thinking_completion = extract_thinking(content)
        reward, step_end_pos = critical_value_reward_regex(thinking_completion, pro) if regex else critical_value_reward(thinking_completion, pro)
        rewards.append(reward)
        steps_end_pos.append(step_end_pos)
    if not silence:
        print('\nThinking rewards:', rewards, '\nCorrect thinking steps end pos:', steps_end_pos)

    return rewards, steps_end_pos
        
def accuracy_thinking_reward(completions, solution, process, silence=False, regex = False, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth and assign partial reward for crucial thinking results."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    steps_end_pos = []
    for content, sol, pro in zip(contents, solution, process):
        # outcome reward
        answer = extract_answer(content)
        gold_parsed, answer_parsed, reward = outcome_reward(answer, sol)
        # process reward
        if reward == 0.0:
            # parse ground truth process
            thinking_completion = extract_thinking(content)
            reward, step_end_pos = critical_value_reward_regex(thinking_completion, pro) if regex else critical_value_reward(thinking_completion, pro)
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

def eval_thinking_reward(completion, solution, process, tag=False, regex = False, silence=False, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth and assign partial reward for crucial thinking results."""
    thinking_completion = extract_thinking(completion) if tag else completion
    reward, step_end_pos = critical_value_reward_regex(thinking_completion, process) if regex else critical_value_reward(thinking_completion, process)
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