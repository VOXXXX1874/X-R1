# Import necessary libraries
import re

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