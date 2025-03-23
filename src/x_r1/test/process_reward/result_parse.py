from math_verify import parse, LatexExtractionConfig, verify

# Read solution from solution.txt
with open("src/x_r1/test/process_reward/solution.md", "r") as file:
    solution = file.read()

parse_result = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()], fallback_mode = 'no_fallback')

print('result of parse', parse_result)

print('result of verify', verify(parse_result, parse_result))