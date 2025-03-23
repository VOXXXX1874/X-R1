import random
import argparse
import json

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--num_of_var", type=int, default=8, help="Number of variables")
    parser.add_argument("--num_of_generation", type=int, default=3, help="Number of generations")
    parser.add_argument("--range", type=int, default=2, help="Range of numbers")
    parser.add_argument("--num_of_questions", type=int, default=7500, help="Number of questions")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    dataset = []

    for _ in range(args.num_of_questions):
        num_of_var = args.num_of_var + random.randint(-args.range, args.range)
        num_of_generation = (args.num_of_generation + random.randint(-args.range, args.range)) * num_of_var

        # Randomly choose number of variables from a to z
        variables = random.sample(range(97, 123), num_of_var)
        variables = [chr(i) for i in variables]
        query = random.choice(variables)

        # initialize the value of variables by random number
        values = [random.randint(-5, 5) for _ in range(num_of_var)]
        values = dict(zip(variables, values))

        prompt = f"Given the initial value of variables and the sequential operations, what is the final value of variable {query}? Notice that '=' means assignment, not equality.\n"
        for var, val in values.items():
            prompt += f"{var} = {val}\n"

        equations = []
        # loop through the number of generations
        for _ in range(num_of_generation):
            # randomly choose the variable
            var_1 = random.choice(variables)
            var_2 = random.choice(variables)
            var_3 = random.choice(variables)
            # randomly choose + or - or * or /
            if values[var_3] != 0:
                op = random.choice(["+", "-", "*", "/"])
            else:
                op = random.choice(["+", "-", "*"])
            # generate the equation
            equation = f"{var_1} = {var_2} {op} {var_3}"
            equations.append(equation)
            # Update the value of the variable
            if op == "+":
                values[var_1] = values[var_2] + values[var_3]
            elif op == "-":
                values[var_1] = values[var_2] - values[var_3]
            elif op == "*":
                values[var_1] = values[var_2] * values[var_3]
            elif op == "/":
                values[var_1] = values[var_2] / values[var_3]

        # Emsemble the prompt
        for _, equation in enumerate(equations):
            prompt += f"{equation}\n"

        # Emsemble the solution
        answer = values[query]

        if (answer == 0 or answer == 1 or answer == 2 or answer == -1 or answer == -2) and random.random() < 0.8:
            continue

        dataset.append({
            "problem": prompt,
            "answer": str(answer)
        })

    # Save the dataset
    with open("src/fol_r1/gsc/GSC.json", "w") as f:
        json.dump(dataset, f, indent=4)


