from math_verify import LatexExtractionConfig, ExprExtractionConfig
import re
from itertools import groupby
from typing import Sequence
from math_verify.parser import *

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

        if extracted_match is not None:
            extracted_predictions.append([extracted_match, str_fallback])

    return extracted_predictions

# Read solution from solution.txt
with open("src/x_r1/test/process_reward/solution.md", "r") as file:
    solution = file.read()

gold_parsed = thinking_parse(
    solution,
    extraction_config=[LatexExtractionConfig()],
)

for parsed_results in gold_parsed:
    if len(parsed_results[1]) > 12:
        print(parsed_results)

