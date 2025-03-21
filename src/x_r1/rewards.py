"""Reward functions for GRPO training."""

import re
from typing import Dict
import os
from openai import OpenAI
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
import math
from math_verify.parser import *

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
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def extract_thinking(text):
    """Extract content between <think> tags."""
    if text is None:
        return ""
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def thinking_parse(
    pred: str,
    extraction_config: Sequence[ExtractionTarget] = [
        LatexExtractionConfig(),
        ExprExtractionConfig(),
    ],
    parsing_timeout: int = 3,
):
    """Extracts and parses all mathematical expressions appearing in a prediction string.
    """
    try:
        target_res = get_extraction_regexes(extraction_config)
        return extract_target_from_pred(pred, target_res, timeout_seconds=parsing_timeout)
    except Exception as e:
        print(f"Error during parsing: {e}")
        return []


def extract_target_from_pred(
    pred: str,
    target_res: list[tuple[list[tuple[re.Pattern[str], int]], ExtractionTarget]],
    timeout_seconds: int,
):
    """Extracts targets from a prediction string using regex patterns.
    Returns all sucesffuly extracted match.
    """
    extracted_predictions = []

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

    # Sort matches by end position (rightmost first) and then by start position (leftmost first)
    matches_with_pos = sorted(
        matches_with_pos, key=lambda x: (x[2], -x[1]), reverse=True
    )

    # Try to extract from each match, starting from rightmost
    for match, _, _, target_type in matches_with_pos:
        extracted_match, str_fallback = extract_match(match, target_type, timeout_seconds=timeout_seconds)

        if extracted_match is not None and len(str_fallback) > 8:
            # check duplicate
            for extracted in extracted_predictions:
                if verify(extracted, extracted_match):
                    break
            else:
                extracted_predictions.append(extracted_match)

    return extracted_predictions

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

# for training
def accuracy_reward(completions, solution, silence=False, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        # First try latex parsing
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            # print('latex gold parsed')
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                extract_answer(content),
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            if len(answer_parsed) == 0:
                answer_parsed = parse(
                    '$' + extract_answer(content)+'$',
                    extraction_mode="first_match",
                    extraction_config=[LatexExtractionConfig()],
                )
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            reward = float(verify(answer_parsed, gold_parsed))
            # print('\nprompt:', prompt)
            if not silence:
                print('-'*100)
                print('\nanswer_parsed:', answer_parsed, '\ngold_parsed:', gold_parsed, '\nreward:', reward)
        else:
            # For medical text answers, extract from <answer> tags and use GPT4O-mini for evaluation
            answer_content = extract_answer(content)
            normalized_content = normalize_text(answer_content)
            normalized_solution = normalize_text(sol)
            reward = evaluate_answer_similarity(normalized_content, normalized_solution)
            if not silence:
                print('-'*100)
                print('\nanswer_parsed:', normalized_content, '\ngold_parsed:', normalized_solution, '\nreward:', reward)
        rewards.append(reward)
    if not silence:
        print('\naccuracy rewards:', rewards)

    return rewards

def accuracy_thinking_reward(completions, solution, process, silence=False, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth and assign partial reward for crucial thinking results."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        # parse ground truth
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        # parse predicted answer
        answer_parsed = parse(
            extract_answer(content),
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(answer_parsed) == 0:
            answer_parsed = parse(
                '$' + extract_answer(content)+'$',
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
        # outcome reward
        reward = float(verify(answer_parsed, gold_parsed))
        # process reward
        if reward == 0.0:
            # parse ground truth process
            gold_thinking = thinking_parse(
                process,
                extraction_config=[LatexExtractionConfig()],
            )
            if len(gold_thinking) != 0:
                # parse predicted process
                thinking_parsed = thinking_parse(
                    extract_thinking(content),
                    extraction_config=[LatexExtractionConfig()],
                )
                atom_reward = 0.8 / len(gold_thinking)
                for gold_thinking in gold_thinking:
                    for thinking in thinking_parsed:
                        if verify(thinking, gold_thinking):
                            print('thinking:', thinking, 'gold_thinking:', gold_thinking)
                            reward += atom_reward
                            break
        if not silence:
            print('-'*100)
            print('\nanswer_parsed:', answer_parsed, '\ngold_parsed:', gold_parsed, '\nreward:', reward)

        rewards.append(reward)
    if not silence:
        print('\naccuracy thinking rewards:', rewards)

    return rewards
        

# for benchmark.py
def eval_answer_reward(completion, answer, tag=False, silence=False, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    '''
    input is completion string, answer is extracted gold answer.
    '''
    gold_parsed = parse(
        answer,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
    completion = extract_answer(completion) if tag else completion
    answer_parsed = parse(
        completion,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
    if len(answer_parsed) == 0 and tag:
        answer_parsed = parse(
            '$' + completion + '$',
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
    reward = float(verify(answer_parsed, gold_parsed))
    if not silence:
        print('-'*100)
        print('\nanswer_parsed:', answer_parsed, '\ngold_parsed:', gold_parsed, '\nreward:', reward)
    return reward

def eval_answer_thinking_reward(completion, answer, process, tag=False, silence=False, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    '''
    input is completion string, answer is extracted gold answer.
    '''
    gold_parsed = parse(
        answer,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
    answer_completion = extract_answer(completion) if tag else completion
    answer_parsed = parse(
        answer_completion,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
    if len(answer_parsed) == 0 and tag:
        answer_parsed = parse(
            '$' + answer_completion + '$',
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
    reward = float(verify(answer_parsed, gold_parsed))
    if reward == 0.0:
        gold_thinking = thinking_parse(
            process,
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_thinking) != 0:
            thinking_completion = extract_thinking(completion) if tag else completion
            thinking_parsed = thinking_parse(
                thinking_completion,
                extraction_config=[LatexExtractionConfig()],
            )
            atom_reward = 0.8 / len(gold_thinking)
            for gold_thinking in gold_thinking:
                for thinking in thinking_parsed:
                    if verify(thinking, gold_thinking):
                        print('thinking:', thinking, 'gold_thinking:', gold_thinking)
                        reward += atom_reward
                        break
    if not silence:
        print('-'*100)
        print('\nanswer_parsed:', answer_parsed, '\ngold_parsed:', gold_parsed, '\nreward:', reward)
    return reward


def format_reward(completions, silence=False, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]

    rewards = [1.0 if match else 0.0 for match in matches]
    if not silence:
        print('-'*100)
        print('\nformat rewards:', rewards)
    return rewards


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
