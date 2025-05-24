from math_verify import parse, LatexExtractionConfig, verify
from sympy import nan, zoo, Eq, oo, Expr, Lt
from sympy.core.relational import Relational
from sympy.logic.boolalg import BooleanTrue, BooleanFalse
import random
import re
import numpy as np
import time

def expression_classification(expression):
    # perform classification on the variables in the expression
    # and construct a dictionary with the variables as keys and expressions as values
    expression_dict = {}
    no_free_symbols_list = []
    for expr_item in expression: # Assuming expression is list of (id, sympy_obj)
        symbols_expr = expr_item[1].free_symbols
        if len(symbols_expr) == 0:
            no_free_symbols_list.append(expr_item)
            continue
        # Convert the set of symbols to a frozenset for immutability and hashing
        symbols_key = frozenset(symbols_expr)
        if symbols_key not in expression_dict:
            expression_dict[symbols_key] = []
        expression_dict[symbols_key].append(expr_item)
        
    return expression_dict, no_free_symbols_list

def majority_voting_expression(expression):
    expression_dict = expression_classification(expression)

    print(expression_dict)
    
    voted_expression = []
    for symbols_key, expr_list in expression_dict.items():
        num_expressions_in_group = len(expr_list)

        if num_expressions_in_group == 1:
            # If only one expression, its agreement score is with itself across all assignments
            agreement_score = 128.0 
            voted_expression.append((expr_list[0], agreement_score))
            continue

        # Perform 128 random assignments and transform each expression into a vector
        expr_vectors = np.zeros((num_expressions_in_group, 128))
        for i in range(128):
            # Corrected random number generation
            random_assignments = {symbol: random.uniform(-100, 100) for symbol in symbols_key}
            
            evaluated_values_for_col_i = []
            try:
                for k_expr_idx, expr_tuple in enumerate(expr_list):
                    val = expr_tuple[1].subs(random_assignments)
                    # Handle SymPy special types before converting to float for numpy array
                    if val == zoo: # SymPy's complex infinity
                        evaluated_values_for_col_i.append(np.nan)
                    elif val == oo: # SymPy's positive infinity
                        evaluated_values_for_col_i.append(np.inf)
                    elif val == -oo: # SymPy's negative infinity
                        evaluated_values_for_col_i.append(-np.inf)
                    elif val == nan: # SymPy's Not a Number
                        evaluated_values_for_col_i.append(np.nan)
                    elif isinstance(val, Expr) and not val.is_number: # Check if it's still symbolic
                        evaluated_values_for_col_i.append(np.nan) # Mark as NaN if not a number
                    else:
                        try:
                            evaluated_values_for_col_i.append(float(val))
                        except (TypeError, ValueError):
                            evaluated_values_for_col_i.append(np.nan) # Mark as NaN if conversion fails
                
                if len(evaluated_values_for_col_i) == num_expressions_in_group:
                     expr_vectors[:, i] = np.array(evaluated_values_for_col_i, dtype=float)
                else: # Should not happen if logic is correct
                    expr_vectors[:, i] = np.nan # Fallback: fill with NaN

            except Exception as e:
                # print(f"Error during substitution for column {i}, assignments {random_assignments}. Error: {e}")
                # Fill the problematic column with NaNs instead of retrying with i -= 1
                expr_vectors[:, i] = np.nan 
                # The loop for i will continue to the next iteration.
        
        # Replace NaN, inf, and -inf with 0 (as per original logic)
        expr_vectors = np.nan_to_num(expr_vectors, nan=0, posinf=0, neginf=0)

        # --- New Majority Voting Logic --- 
        # FIXME: A better algorithm is possible depend on the bottleneck
        agreement_scores = np.zeros(num_expressions_in_group)
        for i_idx in range(num_expressions_in_group):
            total_matches_for_expr_i = 0
            for j_idx in range(num_expressions_in_group):
                # Calculate similarity based on the new criterion
                # (number of assignments where abs difference < 1e-3)
                differences = np.abs(expr_vectors[i_idx, :] - expr_vectors[j_idx, :])
                matches = np.sum(differences < 1e-3)
                total_matches_for_expr_i += matches
            agreement_scores[i_idx] = total_matches_for_expr_i
        
        max_score_idx = np.argmax(agreement_scores)
        voted_expression.append((expr_list[max_score_idx], agreement_scores[max_score_idx]))

    # Sort the voted expression by the new agreement score
    voted_expression = sorted(voted_expression, key=lambda x: x[1], reverse=True)
    
    if len(voted_expression) > 0:
        final_voted_expr = voted_expression[0]
    else:
        final_voted_expr = None
    return final_voted_expr

def majority_voting_relational(expression): # 'expression' here is a list of relational objects
    expression_dict = expression_classification(expression)
    
    voted_results = [] # Stores (relational_object, score)
    for symbols_key, expr_list in expression_dict.items():
        num_relationals_in_group = len(expr_list)

        if num_relationals_in_group == 1:
            # If only one relational, its agreement score is with itself across all assignments
            agreement_score = 128.0 
            voted_results.append((expr_list[0], agreement_score))
            continue

        # Perform 128 random assignments and transform each relational into a boolean vector
        # True/False outcomes from substitutions
        relational_vectors = np.zeros((num_relationals_in_group, 128), dtype=bool) 
        
        for i in range(128): # For each random assignment trial
            random_assignments = {symbol: random.uniform(-100, 100) for symbol in symbols_key}
            
            evaluated_bools_for_col_i = []
            try:
                for k_rel_idx, rel_tuple in enumerate(expr_list):
                    val = rel_tuple[1].subs(random_assignments)
                    if isinstance(val, BooleanTrue):
                        evaluated_bools_for_col_i.append(True)
                    elif isinstance(val, BooleanFalse):
                        evaluated_bools_for_col_i.append(False)
                    else:
                        # Relational did not evaluate to a clear True/False
                        # (e.g., still symbolic, or an error during evaluation not caught below)
                        # Defaulting to False for this trial for this relational.
                        # Consider logging this case if it's unexpected.
                        # print(f"Warning: Relational {rel_tuple[1]} with {random_assignments} evaluated to {val}, not True/False. Using False.")
                        evaluated_bools_for_col_i.append(False) 
                
                if len(evaluated_bools_for_col_i) == num_relationals_in_group:
                     relational_vectors[:, i] = np.array(evaluated_bools_for_col_i, dtype=bool)
                else:
                    # Fallback: fill with False if lengths don't match (should not happen)
                    relational_vectors[:, i] = False 

            except Exception as e:
                # print(f"Error during relational substitution for column {i}, assignments {random_assignments}. Error: {e}")
                # Fill the problematic column with False
                relational_vectors[:, i] = False 
        
        # Calculate agreement scores based on vector equality
        agreement_scores = np.zeros(num_relationals_in_group)
        for i_idx in range(num_relationals_in_group):
            total_matches_for_rel_i = 0
            for j_idx in range(num_relationals_in_group):
                # Number of positions where the boolean vectors are identical
                matches = np.sum(relational_vectors[i_idx, :] == relational_vectors[j_idx, :])
                total_matches_for_rel_i += matches
            agreement_scores[i_idx] = total_matches_for_rel_i
        
        max_score_idx = np.argmax(agreement_scores)
        voted_results.append((expr_list[max_score_idx], agreement_scores[max_score_idx]))

    # Sort the voted relationals by their agreement score
    voted_results = sorted(voted_results, key=lambda x: x[1], reverse=True)
    
    if len(voted_results) > 0:
        final_voted_relational = voted_results[0] # Return the relational object itself
    else:
        final_voted_relational = None
    return final_voted_relational

def majority_voting_equation(expression):
    # Convert the expression list from equation to relational by subsituting '=' with '<'
    for i in range(len(expression)):
        expression[i] = (expression[i][0], Lt(expression[i][1].lhs, expression[i][1].rhs))
    # Call the majority voting relational function
    voted_relational = majority_voting_relational(expression)
    # Convert the voted relational back to equation
    if voted_relational is not None:
        voted_equation = ((voted_relational[0][0], Eq(voted_relational[0][1].lhs, voted_relational[0][1].rhs)), voted_relational[1])
    else:
        voted_equation = None
    return voted_equation

# This script is used to test the majority voting algorithm on the file voting.txt
start_time = time.time()

# load the file voting.txt
with open("src/x_r1/test/MVOT/one_quesiton/voting.txt", "r") as file:
    voting = file.read()

# parse the voting with "<answer>(.*)</answer>"
pattern = r"<answer>(.*?)</answer>"
matches = re.findall(pattern, voting, re.DOTALL)
# Collect all the parsed answers
parsed_answers = []
for i, match in enumerate(matches):
    # Remove any leading or trailing whitespace
    cleaned_match = match.strip()
    # Append the cleaned match to the list
    parsed_answers.append((i,parse(cleaned_match, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()], fallback_mode = 'no_fallback')))

print("time taken to read and parse the answers:", time.time() - start_time)
start_time = time.time()

# Remove the empty parsed answers []
parsed_answers = [(answer[0], answer[1][0]) for answer in parsed_answers if answer[1] != []] 

# classify the sympy answers into expression, relational, and equation
expression = []
relational = []
equation = []
for parsed_answer in parsed_answers:
    if isinstance(parsed_answer[1], Eq):
        equation.append(parsed_answer)
    elif isinstance(parsed_answer[1], Relational):
        relational.append(parsed_answer)
    elif isinstance(parsed_answer[1], Expr):
        expression.append(parsed_answer)
    else:
        # FIXME: How to deal with set?
        pass

voted_expression = majority_voting_expression(expression)
print('voted_expression', voted_expression)
print('time taken to vote the expression:', time.time() - start_time)
start_time = time.time()

voted_relational = majority_voting_relational(relational)
print('voted_relational', voted_relational)
print('time taken to vote the relational:', time.time() - start_time)
start_time = time.time()

voted_equation = majority_voting_equation(equation)
print('voted_equation', voted_equation)
print('time taken to vote the equation:', time.time() - start_time)