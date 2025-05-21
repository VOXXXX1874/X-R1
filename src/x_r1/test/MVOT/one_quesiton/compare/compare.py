from math_verify import parse, LatexExtractionConfig, verify
from sympy import nan, zoo
from sympy import Lt
from sympy.core.relational import Relational
import random
    
def relational_compare(expr1, expr2):
    """
    Compare two expressions.
    """
    # Check if both expressions are relational
    if not isinstance(expr1, Relational) or not isinstance(expr2, Relational):
        raise ValueError("Both expressions must be relational")

    symbols_expr1 = expr1.free_symbols
    symbols_expr2 = expr2.free_symbols
    if len(symbols_expr1) != len(symbols_expr2):
        return False
    
    # Perform 10 random assignments
    for _ in range(10):
        random_assignments = {symbol: random.randint(-100, 100) for symbol in symbols_expr1}
        # Evaluate both expressions with the random assignments
        expr1_evaluated = expr1.subs(random_assignments)
        expr2_evaluated = expr2.subs(random_assignments)
        
        # Compare the evaluated expressions
        # error tolerance 1e-3
        if verify(expr1_evaluated, expr2_evaluated, numeric_precision=1e-5) == False:
            print('expr1_evaluated', expr1_evaluated)
            print('expr2_evaluated', expr2_evaluated)
            print('random_assignments', random_assignments)
            return False
        
    return True

# Read solution from solution.txt
with open("src/x_r1/test/MVOT/one_quesiton/compare/solution.md", "r") as file:
    solution = file.read()

# Read solution_pm from solution_pm.txt
with open("src/x_r1/test/MVOT/one_quesiton/compare/solution_pm.md", "r") as file:
    solution_pm = file.read()

# parse the solution
parse_result = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()], fallback_mode = 'no_fallback')
print('result of parse', parse_result)

# if the parse result is NaN or zoo
if parse_result[0] == nan or parse_result[0] == zoo:
    print('result of verify', False)
else:
    pm_parse_result = parse(solution_pm, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()], fallback_mode = 'no_fallback')
    print('result of parse_pm', pm_parse_result)

    # Compare the two expressions
    result = relational_compare(parse_result[0], pm_parse_result[0])
    print('result of verify', result)
    

    