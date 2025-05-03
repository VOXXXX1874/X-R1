import re
import time

def extract_thinking(text):
    """Extract content between <think> tags."""
    if text is None:
        return ""
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def response_thinking_parse(text):
    """Parse the response text to decompose the thinking part."""
    # separate the response into lines
    lines = text.split("\n")
    # remove leading and trailing whitespace from each line
    lines = [line.strip() for line in lines]
    # remove empty lines
    lines = [line for line in lines if line]
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

def gt_thinking_parse(text):
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

# Read solution from solution.txt
with open("src/x_r1/test/reward/solution.md", "r") as file:
    solution = file.read()

# Read solution from solution.txt
with open("src/x_r1/test/reward/solution_pm.md", "r") as file:
    solution_pm = file.read()

# Count the time taken to parse and verify the solution
start_time = time.time()

response_parsed, response_end_pos = response_thinking_parse(
    extract_thinking(solution),
)

gt_parsed, _ = gt_thinking_parse(
    extract_thinking(solution_pm),
)

print("response_parsed", response_parsed, "response_end_pos", response_end_pos)
print("gt_parsed", gt_parsed[0])

atomic_reward = 1.0/len(gt_parsed)
reward = 0.0
final_end_pos = 0
for gt_element in gt_parsed:
    for response_element, end_pos in zip(response_parsed, response_end_pos):
        if regex_verify(gt_element, response_element):
            print("reponse_parsed_results", response_element)
            print("gt_parsed_results", gt_element)
            print("verified")
            reward += atomic_reward
            final_end_pos = end_pos
            break

print("--- %s seconds ---" % (time.time() - start_time))