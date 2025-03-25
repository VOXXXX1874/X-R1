from math_verify import parse, LatexExtractionConfig, verify
from sympy import nan, zoo

# Read solution from solution.txt
with open("src/x_r1/test/reward/solution.md", "r") as file:
    solution = file.read()

# Read solution_pm from solution_pm.txt
with open("src/x_r1/test/reward/solution_pm.md", "r") as file:
    solution_pm = file.read()

parse_result = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()], fallback_mode = 'no_fallback')

print('result of parse', parse_result)

# if the parse result is NaN or zoo
if parse_result[0] == nan or parse_result[0] == zoo:
    print('result of verify', False)
else:

    pm_parse_result = parse(solution_pm, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()], fallback_mode = 'no_fallback')

    print('result of parse_pm', pm_parse_result)

    print('result of verify', verify(parse_result, pm_parse_result))