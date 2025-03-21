from math_verify import parse, LatexExtractionConfig

# Read solution from solution.txt
with open("src/x_r1/test/process_reward/solution.md", "r") as file:
    solution = file.read()

print('result of parse', parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()], fallback_mode = 'no_fallback'))