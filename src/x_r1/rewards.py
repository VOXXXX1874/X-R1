"""Reward functions for GRPO training."""

import re
from typing import Dict
import os
from openai import OpenAI
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
import math
from math_verify.parser import *
from sympy import nan, zoo

# Initialize OpenAI client
client = None

def normalize_text(text):
    """Normalize text by removing extra whitespace, converting to lowercase."""
    if text is None:
        return ""
    # Remove extra whitespace and convert to lowercase
    text = re.sub(r'\s+', ' ', text.strip().lower())
    return text

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


def evaluate_answer_similarity(answer, solution):
    """Use GPT4O-mini to evaluate answer similarity."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical answer evaluator. Compare the student's answer with the correct solution and output ONLY '1.0' if they match in meaning, or '0.0' if they don't match. No other output is allowed."
                },
                {
                    "role": "user",
                    "content": f"Student answer: {answer}\nCorrect solution: {solution}\nOutput only 1.0 or 0.0:"
                }
            ],
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        return float(result)
    except Exception as e:
        print(f"Error in GPT evaluation: {e}")
        # If API call fails, fall back to simple text matching
        return 1.0 if normalize_text(answer) == normalize_text(solution) else 0.0
    
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
    # replace '=' with '.*' 
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


def reasoning_steps_reward(completions, **kwargs):
    """Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic nubmer 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solutions: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solutions: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solutions):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward
