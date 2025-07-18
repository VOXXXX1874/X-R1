# Import libraries to read json files and manipulate data
import json
# regular expression library for pattern matching
from math_verify.parser import get_extraction_regexes
import re
from math_verify import LatexExtractionConfig, parse, verify
from math_verify.parser import *
import numpy as np

class expression_node:
    """
    A class to represent an expression as a node in a tree structure.
    The node contains numbers, operators, and phrases extracted from a mathematical expression.
    It also contains the end position and links to other nodes if needed.
    """
    def __init__(self, numbers, operators, phrases):
        self.numbers = numbers
        self.operators = operators
        self.phrases = phrases
        self.state_value = 0  # Placeholder for the value of the expression
        self.links = []  # Links to other nodes can be added later

    def __repr__(self):
        return f"ExpressionNode(numbers={self.numbers}, operators={self.operators}, phrases={self.phrases} links={self.links})"

    def similarity(self, another_node):
        """
        Computes the similarity between this node and another expression_node.
        The similarity is based on the number of common elements in numbers, operators, and phrases.
        """
        total_numbers = min(len(self.numbers), len(another_node.numbers)) + 1e-1
        common_numbers = len(self.numbers.intersection(another_node.numbers))
        incommon_numbers = len(self.numbers - another_node.numbers) + len(another_node.numbers - self.numbers)
        similarity_numbers = (common_numbers - incommon_numbers) / total_numbers
        
        total_operators = min(len(self.operators), len(another_node.operators)) + 1e-1
        common_operators = len(self.operators.intersection(another_node.operators))
        incommon_operators = len(self.operators - another_node.operators) + len(another_node.operators - self.operators)
        similarity_operators = (common_operators - incommon_operators) / total_operators

        total_phrases = min(len(self.phrases), len(another_node.phrases)) + 1e-1
        common_phrases = len(self.phrases.intersection(another_node.phrases))
        incommon_phrases = len(self.phrases - another_node.phrases) + len(another_node.phrases - self.phrases)
        similarity_phrases = (common_phrases - incommon_phrases) / total_phrases

        return min(similarity_numbers, similarity_operators, similarity_phrases)
    
    def update_link(self, node_index, num_pass):
        """
        Updates the link to another node based on the node index and the number of passes.
        This method can be used to establish connections between nodes in the tree.
        """
        for link in self.links:
            if link['node_index'] == node_index:
                link['num_pass'] += num_pass
                return
        else:
            self.links.append({'node_index': node_index, 'num_pass': num_pass})

    def merge(self, another_node):
        """
        Merges another expression_node into this node.
        This method combines the numbers, operators, and phrases from both nodes.
        """
        self.numbers.update(another_node.numbers)
        self.operators.update(another_node.operators)
        self.phrases.update(another_node.phrases)
        
    
class MDP_tree:
    """
    A class to represent a tree structure for MDP (Markov Decision Process) in solving math problem.
    The tree contains nodes that represent mathematical expressions.
    """
    def __init__(self, expressions_list, final_reward):
        """
        Initializes the MDP_tree with a list of expressions and their end positions.
        Each expression is parsed into an expression_node.
        """
        self.root_node = expression_node(numbers=set(), operators=set(), phrases=set())
        self.node_list = [self.root_node]
        self.wrong_state = -1
        self.correct_state = -1
        self.update(expressions_list, final_reward)

    def __repr__(self):
        """
        Returns a string representation of the MDP_tree.
        """
        return f"MDP_tree with {len(self.node_list)} nodes: {[str(node) for node in self.node_list]}"
    
    def update(self, new_expressions_list, final_reward=0):
        """
        Updates the MDP_tree with new expressions and their end positions.
        This method can be used to add more nodes to the existing tree.
        """
        # First choose the expressions that unique enough to be a node
        all_nodes = [expression_node(**expression_parse(expression)) for expression in new_expressions_list]
        similarities = np.array([[node.similarity(other_node) for other_node in all_nodes] for node in all_nodes])
        # Select 3 nodes with the lowest similarity to each other
        selected_nodes_1 = np.argmin(np.sum(similarities, axis=1)[:len(all_nodes)//3])
        selected_nodes_2 = np.argmin(np.sum(similarities, axis=1)[len(all_nodes)//3:2*len(all_nodes)//3]) + len(all_nodes)//3
        selected_nodes_3 = np.argmin(np.sum(similarities, axis=1)[2*len(all_nodes)//3:-1]) + 2*len(all_nodes)//3
        selected_nodes = [all_nodes[selected_nodes_1], all_nodes[selected_nodes_2], all_nodes[selected_nodes_3]]
        # Parse the new expressions and create nodes
        last_node_index = 0
        for i, node in enumerate(selected_nodes):
            if i == len(selected_nodes) - 1:
                if final_reward < 0.5:
                    if self.wrong_state == -1:
                        self.node_list[last_node_index].update_link(len(self.node_list), 1)
                        self.wrong_state = len(self.node_list)
                        self.node_list.append(node)
                    else:
                        self.node_list[last_node_index].update_link(self.wrong_state, 1)
                    self.node_list[self.wrong_state].merge(node)
                    node.state_value = final_reward  # Set the last node's value to final_reward (final state)
                else:
                    if self.correct_state == -1:
                        self.node_list[last_node_index].update_link(len(self.node_list), 1)
                        self.correct_state = len(self.node_list)
                        self.node_list.append(node)
                    else:
                        self.node_list[last_node_index].update_link(self.correct_state, 1)
                    self.node_list[self.correct_state].merge(node)
                    node.state_value = final_reward
            else:
                # Search for similar nodes in the existing tree to merge
                similar_node_index = -1
                for i, existing_node in enumerate(self.node_list[1:], start=1):
                    if existing_node.similarity(node) > 0.6:
                        similar_node_index = i
                        break
                # If a similar node is found, merge it; otherwise, add a new node
                if similar_node_index != -1:
                    if last_node_index != similar_node_index:
                        self.node_list[last_node_index].update_link(similar_node_index, 1)
                    last_node_index = similar_node_index
                    self.node_list[similar_node_index].merge(node)
                else:
                    self.node_list[last_node_index].update_link(len(self.node_list), 1)
                    self.node_list.append(node)
                    last_node_index = len(self.node_list) - 1


def expression_parse(
    pred: str,
):
    """
    Parses the number, operator, and phrase in a latex expression.
    """
    # Define regex patterns for numbers, operators, and phrases
    number_pattern = re.compile(r'\d+')
    operator_pattern = re.compile(r'[+\-*/]')
    phrase_pattern = re.compile(r'[a-zA-Z]+')

    # Find all matches in the prediction string
    numbers = number_pattern.findall(pred)
    operators = operator_pattern.findall(pred)
    phrases = phrase_pattern.findall(pred)

    # Combine the matches into a tuple
    parsed_expression = {
        'numbers': set(numbers),
        'operators': set(operators),
        'phrases': set(phrases)
    }

    return parsed_expression

def thinking_parse(
    pred: str,
    extraction_config: Sequence[ExtractionTarget] = [
        LatexExtractionConfig(),
        ExprExtractionConfig(),
    ],
    parsing_timeout: int = 1,
):
    """
    Parses all mathematical expressions appearing in a prediction string.
    Extract possible 'critical value' from the expression.
    """
    try:
        target_res = get_extraction_regexes(extraction_config)
        return timeout(timeout_seconds=parsing_timeout)(extract_target_from_pred)(
            pred,
            target_res,
        )
    except Exception as e:
        # Log the exception if any error occurs during parsing
        logger.exception(f"Error parsing: {pred}, error: {e}")
        return []
    except TimeoutException:
        logger.error(f"Timeout during parsing: {pred}")
        return []


def extract_target_from_pred(
    pred: str,
    target_res: list[tuple[list[tuple[re.Pattern[str], int]], ExtractionTarget]],
):
    """Extracts targets from a prediction string using regex patterns.
    Returns all sucesffuly extracted match.
    """
    extracted_predictions = []
    predictions_end_pos = []
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
    for match, _, end_position, target_type in matches_with_pos:

        extracted_predictions.append(match.group(0))
        predictions_end_pos.append(end_position)

    return extracted_predictions, predictions_end_pos

# Test the thinking_parse function
states, positions = thinking_parse(
            "To solve the given problem, we start with the equation:\n\n\\[\n\\frac{a}{25-a} + \\frac{b}{65-b} + \\frac{c}{60-c} = 7\n\\]\n\nWe need to evaluate the expression:\n\n\\[\n\\frac{5}{25-a} + \\frac{13}{65-b} + \\frac{12}{60-c}\n\\]\n\nFirst, let's make a substitution to simplify the problem. Let \\( x = 25 - a \\). Then, \\( a = 25 - x \\). Similarly, let \\( y = 65 - b \\), so \\( b = 65 - y \\), and let \\( z = 60 - c \\), so \\( c = 60 - z \\). Substituting these into the original equation, we get:\n\n\\[\n\\frac{a}{25-a} + \\frac{b}{65-b} + \\frac{c}{60-c} = \\frac{25-x}{x} + \\frac{65-y}{y} + \\frac{60-z}{z} = 7\n\\]\n\nThis can be rewritten as:\n\n\\[\n\\frac{25}{x} - 1 + \\frac{65}{y} - 1 + \\frac{60}{z} - 1 = 7\n\\]\n\nSimplifying the left side, we get:\n\n\\[\n\\frac{25}{x} + \\frac{65}{y} + \\frac{60}{z} - 3 = 7\n\\]\n\nAdding 3 to both sides, we obtain:\n\n\\[\n\\frac{25}{x} + \\frac{65}{y} + \\frac{60}{z} = 10\n\\]\n\nNow, we need to find the value of:\n\n\\[\n\\frac{5}{25-a} + \\frac{13}{65-b} + \\frac{12}{60-c}\n\\]\n\nSubstituting back \\( x = 25 - a \\), \\( y = 65 - b \\), and \\( z = 60 - c \\), we get:\n\n\\[\n\\frac{5}{x} + \\frac{13}{y} + \\frac{12}{z}\n\\]\n\nSince we already know that:\n\n\\[\n\\frac{25}{x} + \\frac{65}{y} + \\frac{60}{z} = 10\n\\]\n\nWe can conclude that:\n\n\\[\n\\frac{5}{x} + \\frac{13}{y} + \\frac{12}{z} = \\frac{1}{2} \\left( \\frac{25}{x} + \\frac{65}{y} + \\frac{60}{z} \\right) = \\frac{1}{2} \\times 10 = 5\n\\]\n\nTherefore, the value of the expression is:\n\n\\[\n\\boxed{5}\n\\]",
            extraction_config=[LatexExtractionConfig()],
        )

# Construct the MDP tree with the parsed expressions
mdp_tree = MDP_tree(states, 0)
## Update the MDP tree with new response
#states, positions = thinking_parse(
#            "To solve the given problem, we need to evaluate the expression \\(\\frac{5}{25-a} + \\frac{13}{65-b} + \\frac{12}{60-c}\\) given that \\(\\frac{a}{25-a} + \\frac{b}{65-b} + \\frac{c}{60-c} = 7\\). Let's denote \\(x = \\frac{a}{25-a}\\), \\(y = \\frac{b}{65-b}\\), and \\(z = \\frac{c}{60-c}\\). Then we have:\n\n\\[x + y + z = 7.\\]\n\nWe need to find the value of \\(5x + 13y + 12z\\). To do this, we can use the fact that \\(y = \\frac{1}{x^*}\\) and \\(z = \\frac{1}{x^* - 1}\\) where \\(x^*\\) is the value of the expression \\(\\frac{25-a}{a}\\) when \\(a = \\frac{1}{x}\\).\n\nFirst, let's express \\(y\\) and \\(z\\) in a different form. Notice that:\n\n\\[y = \\frac{1}{x^*} = \\frac{1}{\\frac{25-a}{a}} = \\frac{a}{25-a},\\]\n\\[z = \\frac{1}{x^* - 1} = \\frac{1}{\\frac{25-a}{a} - 1} = \\frac{1}{\\frac{25-a-a}{a}} = \\frac{a}{25-2a}.\\]\n\nNow, let's substitute \\(x = \\frac{a}{25-a}\\), \\(y = \\frac{b}{65-b}\\), and \\(z = \\frac{c}{60-c}\\) into the expression we need to evaluate:\n\n\\[5x + 13y + 12z = 5 \\left(\\frac{a}{25-a}\\right) + 13 \\left(\\frac{b}{65-b}\\right) + 12 \\left(\\frac{c}{60-c}\\right).\\]\n\nFrom the given condition, we know that \\(\\frac{a}{25-a} + \\frac{b}{65-b} + \\frac{c}{60-c} = 7\\). We can use this to help us find the value of the expression \\(5 \\left(\\frac{a}{25-a}\\right) + 13 \\left(\\frac{b}{65-b}\\right) + 12 \\left(\\frac{c}{60-c}\\right)\\).\n\nNotice that:\n\n\\[5 \\left(\\frac{a}{25-a}\\right) + 13 \\left(\\frac{b}{65-b}\\right) + 12 \\left(\\frac{c}{60-c}\\right) = 5 \\left(\\frac{1}{7-1} + \\frac{1}{65-63} + \\frac{1}{60-59}\\right) + 13 \\left(\\frac{1}{7-1} + \\frac{1}{65-63} + \\frac{1}{60-59}\\right).\\]\n\nSince \\(\\frac{a}{25-a} + \\frac{b}{65-b} + \\frac{c}{60-c} = 7\\), the expression simplifies to:\n\n\\[5 \\left(\\frac{1}{6} + \\frac{1}{2} + 1\\right) + 13 \\left(\\frac{1}{6} + \\frac{1}{2} + 1\\right) = 7 \\left(\\frac{1}{6} + \\frac{1}{2} + 1\\right) = 7 \\left(\\frac{1 + 3 + 6}{6}\\right) = 7 \\left(\\frac{10}{6}\\right) = 7 \\left(\\frac{5}{3}\\right) = \\frac{35}{3}.\\]\n\nThus, the value of the expression is:\n\n\\[\n\\boxed{\\frac{35}{3}}.\n\\]",
#            extraction_config=[LatexExtractionConfig()],
#        )
## Update the MDP tree with the new response
#mdp_tree.update(states, 1)
# Print the MDP tree to see the structure
print(mdp_tree)
exit()

# Read the json file train.json
with open("XR1-7500/extend/train.json", "r") as f:
    math_qa_dataset = json.load(f)

# Initialize an empty list to store the processed data
MDP_tree_math_qa_dataset = []

# Iterate through each item in the dataset
for item in math_qa_dataset:
    # Extract the solution, correct_responses, and wrong_responses
    solution = item.get("solution", "")
    correct_responses = item.get("correct_responses", [])
    wrong_responses = item.get("wrong_responses", [])
    # Create a new MDP_tree for the solution
    solution_states, solution_positions = thinking_parse(
        solution,
        extraction_config=[LatexExtractionConfig()],
    )
    problem_mdp_tree = MDP_tree(solution_states, 1)
    # Update this MDP_tree with correct responses
    for correct_response in correct_responses:
        correct_states, correct_positions = thinking_parse(
            correct_response,
            extraction_config=[LatexExtractionConfig()],
        )
        problem_mdp_tree.update(correct_states, 1)
    # Update this MDP_tree with wrong responses
    for wrong_response in wrong_responses:
        wrong_states, wrong_positions = thinking_parse(
            wrong_response,
            extraction_config=[LatexExtractionConfig()],
        )
        problem_mdp_tree.update(wrong_states, 0)

    # print the MDP_tree for the problem
    print(problem_mdp_tree)
    break
        
    