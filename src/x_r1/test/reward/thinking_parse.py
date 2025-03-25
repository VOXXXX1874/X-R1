from math_verify import LatexExtractionConfig, ExprExtractionConfig, verify
import re
from itertools import groupby
from typing import Sequence
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

def thinking_parse(
    pred: str,
    extraction_config: Sequence[ExtractionTarget] = [
        LatexExtractionConfig(),
        ExprExtractionConfig(),
    ],
):
    """Extracts and parses all mathematical expressions appearing in a prediction string.
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
    for match, _, _, target_type in matches_with_pos:
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

    return extracted_predictions

# Read solution from solution.txt
with open("src/x_r1/test/reward/solution.md", "r") as file:
    solution = file.read()

# Read solution from solution.txt
with open("src/x_r1/test/reward/solution_pm.md", "r") as file:
    solution_pm = file.read()

# Count the time taken to parse and verify the solution
start_time = time.time()

gold_parsed = thinking_parse(
    extract_thinking(solution),
    extraction_config=[LatexExtractionConfig()],
)

pm_gold_parsed = thinking_parse(
    extract_thinking(solution_pm),
    extraction_config=[LatexExtractionConfig()],
)

#print("gold_parsed", gold_parsed)
#print("pm_gold_parsed", pm_gold_parsed)

for parsed_results in gold_parsed:
    for pm_parsed_results in pm_gold_parsed:
        if verify(parsed_results, pm_parsed_results):
            print("parsed_results", parsed_results)
            print("pm_parsed_results", pm_parsed_results)
            print("verified")
            break

print("--- %s seconds ---" % (time.time() - start_time))

