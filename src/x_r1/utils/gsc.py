import re
from math_verify import LatexExtractionConfig, parse, verify
from math_verify.parser import *
from sympy import nan, zoo

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

def critical_value_reward_gsc(thinking_completion, process):
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