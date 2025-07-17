# Import libraries to read json files and manipulate data
import json
# regular expression library for pattern matching
import re
from math_verify import LatexExtractionConfig
from math_verify.parser import *
from copy import deepcopy

class number_node:
    """
    A class to represent a state as sequence of numbers
    The node contains existing numbers and new number
    It also contains value of this state and links to other states.
    """
    def __init__(self, existing_numbers, new_number):
        self.existing_numbers = existing_numbers
        self.new_number = new_number
        self.state_value = 0  # Placeholder for the value of the expression
        self.links = []  # Links to other nodes can be added later

    def __repr__(self):
        return f"number_node(existing_numbers={self.existing_numbers}, new_numbers={self.new_number}, links={self.links})"
    
    def similarity(self, another_node):
        """
        Calculates the similarity between the existing numbers of this node and another number_node.
        Returns a float value representing the similarity.
        """
        this_node_number_set = self.existing_numbers.union({self.new_number}) if self.new_number else self.existing_numbers
        another_node_number_set = another_node.existing_numbers.union({another_node.new_number}) if another_node.new_number else another_node.existing_numbers
        # Calculate the similarity based on the existing numbers
        common_numbers = this_node_number_set.intersection(another_node_number_set)
        total_numbers = this_node_number_set.union(another_node_number_set)
        similarity_score = len(common_numbers) / len(total_numbers) if total_numbers else 0
        return similarity_score
    
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
        self.existing_numbers.update(another_node.existing_numbers)
        
    
class MDP_tree:
    """
    A class to represent a tree structure for MDP (Markov Decision Process) in solving math problem.
    The tree contains nodes that represent mathematical expressions.
    """
    def __init__(self, expression_list, final_reward):
        """
        Initializes the MDP_tree with a list of expressions and their end positions.
        Each expression is parsed into an expression_node.
        """
        self.root_node = number_node(existing_numbers=set(), new_number=None)
        self.node_list = [self.root_node]
        self.wrong_state = -1
        self.correct_state = -1
        self.update(expression_list, final_reward)

    def __repr__(self):
        """
        Returns a string representation of the MDP_tree.
        """
        sep = ": "
        return f"MDP_tree with {len(self.node_list)} nodes: {[str(i) + sep + str(node) for i, node in enumerate(self.node_list)]}"
    
    def update(self, expression_list, final_reward=0):
        """
        Updates the MDP_tree with new expressions and their end positions.
        This method can be used to add more nodes to the existing tree.
        """
        # Parse the numbers in the expression_list
        numbers = []
        for expression in expression_list:
            # Extract numbers from the expression using regex
            numbers.extend(number_parse(expression))
        # Parse the new numbers and create nodes
        last_node_index = 0
        for i, number in enumerate(numbers):
            # if the new number is not new, continue
            if number in self.node_list[last_node_index].existing_numbers or number == self.node_list[last_node_index].new_number:
                continue
            # Create a new node with the existing numbers and the new number
            existing_numbers = deepcopy(self.node_list[last_node_index].existing_numbers)
            if self.node_list[last_node_index].new_number:
                existing_numbers.add(self.node_list[last_node_index].new_number)
            node = number_node(existing_numbers=existing_numbers, new_number=number)
            # Search for nodes that share the same new number
            similar_node_index = -1
            for i, existing_node in enumerate(self.node_list[1:], start=1):
                if existing_node.similarity(node) > 0.999:
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

        if final_reward < 0.5:
            if self.wrong_state == -1:
                node = number_node(existing_numbers=set(), new_number=None)
                node.state_value = final_reward  # Set the last node's value to final_reward (final state)
                self.node_list[last_node_index].update_link(len(self.node_list), 1)
                self.wrong_state = len(self.node_list)
                self.node_list.append(node)
            else:
                self.node_list[last_node_index].update_link(self.wrong_state, 1)
        else:
            if self.correct_state == -1:
                node = number_node(existing_numbers=set(), new_number=None)
                node.state_value = final_reward # Set the last node's value to final_reward (final state)
                self.node_list[last_node_index].update_link(len(self.node_list), 1)
                self.correct_state = len(self.node_list)
                self.node_list.append(node)
            else:
                self.node_list[last_node_index].update_link(self.correct_state, 1)

    def trim(self):
        """
        Trims the MDP_tree by removing nodes that has only one pass.
        This method performs a depth-first search (DFS) to identify and remove nodes recursively.
        """
        def dfs(node_index, parent_index):
            """
            Depth-first search to trim the MDP_tree.
            """
            node = self.node_list[node_index]
            # If the node has only one link, remove it
            if len(node.links) == 1 and node.links[0]['num_pass'] == 1:
                # Remove the link from the parent node
                if parent_index != -1:
                    parent_node = self.node_list[parent_index]
                    parent_node.links = [link for link in parent_node.links if link['node_index'] != node_index]
                # Remove the node from the tree
                self.node_list[node_index] = None
                return True
            else:
                # Recursively check all links
                for link in node.links:
                    if dfs(link['node_index'], node_index):
                        # If a child was removed, update the links
                        node.links = [l for l in node.links if l['node_index'] != link['node_index']]
            return False

        # Start DFS from the root node
        dfs(0, -1)
        # Remove None nodes from the list
        self.node_list = [node for node in self.node_list if node is not None]
        

def number_parse(
    pred: str,
):
    """
    Parses the numbers in a response.
    """
    # Define regex patterns for numbers, operators, and phrases
    number_pattern = re.compile(r'\d+')
    # Find all matches in the prediction string before the "\boxed" part
    pred = pred.split("\\boxed")[0]
    numbers = number_pattern.findall(pred)
    # Convert the matches to list of integers
    numbers = [int(num) for num in numbers]
    return numbers

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

## Test the MDP tree with a simple example
## Construct the MDP tree with the response
#states, positions = thinking_parse(
#            "To solve the given problem, we start with the equation:\n\n\\[\n\\frac{a}{25-a} + \\frac{b}{65-b} + \\frac{c}{60-c} = 7\n\\]\n\nWe need to evaluate the expression:\n\n\\[\n\\frac{5}{25-a} + \\frac{13}{65-b} + \\frac{12}{60-c}\n\\]\n\nFirst, let's make a substitution to simplify the problem. Let \\( x = 25 - a \\). Then, \\( a = 25 - x \\). Similarly, let \\( y = 65 - b \\), so \\( b = 65 - y \\), and let \\( z = 60 - c \\), so \\( c = 60 - z \\). Substituting these into the original equation, we get:\n\n\\[\n\\frac{a}{25-a} + \\frac{b}{65-b} + \\frac{c}{60-c} = \\frac{25-x}{x} + \\frac{65-y}{y} + \\frac{60-z}{z} = 7\n\\]\n\nThis can be rewritten as:\n\n\\[\n\\frac{25}{x} - 1 + \\frac{65}{y} - 1 + \\frac{60}{z} - 1 = 7\n\\]\n\nSimplifying the left side, we get:\n\n\\[\n\\frac{25}{x} + \\frac{65}{y} + \\frac{60}{z} - 3 = 7\n\\]\n\nAdding 3 to both sides, we obtain:\n\n\\[\n\\frac{25}{x} + \\frac{65}{y} + \\frac{60}{z} = 10\n\\]\n\nNow, we need to find the value of:\n\n\\[\n\\frac{5}{25-a} + \\frac{13}{65-b} + \\frac{12}{60-c}\n\\]\n\nSubstituting back \\( x = 25 - a \\), \\( y = 65 - b \\), and \\( z = 60 - c \\), we get:\n\n\\[\n\\frac{5}{x} + \\frac{13}{y} + \\frac{12}{z}\n\\]\n\nSince we already know that:\n\n\\[\n\\frac{25}{x} + \\frac{65}{y} + \\frac{60}{z} = 10\n\\]\n\nWe can conclude that:\n\n\\[\n\\frac{5}{x} + \\frac{13}{y} + \\frac{12}{z} = \\frac{1}{2} \\left( \\frac{25}{x} + \\frac{65}{y} + \\frac{60}{z} \\right) = \\frac{1}{2} \\times 10 = 5\n\\]\n\nTherefore, the value of the expression is:\n\n\\[\n\\boxed{5}\n\\]",
#            extraction_config=[LatexExtractionConfig()],
#        )
#mdp_tree = MDP_tree(states, 0)
## Update the MDP tree with new response
#states, positions = thinking_parse(
#            "To solve the given problem, we need to evaluate the expression \\(\\frac{5}{25-a} + \\frac{13}{65-b} + \\frac{12}{60-c}\\) given that \\(\\frac{a}{25-a} + \\frac{b}{65-b} + \\frac{c}{60-c} = 7\\). Let's denote \\(x = \\frac{a}{25-a}\\), \\(y = \\frac{b}{65-b}\\), and \\(z = \\frac{c}{60-c}\\). Then we have:\n\n\\[x + y + z = 7.\\]\n\nWe need to find the value of \\(5x + 13y + 12z\\). To do this, we can use the fact that \\(y = \\frac{1}{x^*}\\) and \\(z = \\frac{1}{x^* - 1}\\) where \\(x^*\\) is the value of the expression \\(\\frac{25-a}{a}\\) when \\(a = \\frac{1}{x}\\).\n\nFirst, let's express \\(y\\) and \\(z\\) in a different form. Notice that:\n\n\\[y = \\frac{1}{x^*} = \\frac{1}{\\frac{25-a}{a}} = \\frac{a}{25-a},\\]\n\\[z = \\frac{1}{x^* - 1} = \\frac{1}{\\frac{25-a}{a} - 1} = \\frac{1}{\\frac{25-a-a}{a}} = \\frac{a}{25-2a}.\\]\n\nNow, let's substitute \\(x = \\frac{a}{25-a}\\), \\(y = \\frac{b}{65-b}\\), and \\(z = \\frac{c}{60-c}\\) into the expression we need to evaluate:\n\n\\[5x + 13y + 12z = 5 \\left(\\frac{a}{25-a}\\right) + 13 \\left(\\frac{b}{65-b}\\right) + 12 \\left(\\frac{c}{60-c}\\right).\\]\n\nFrom the given condition, we know that \\(\\frac{a}{25-a} + \\frac{b}{65-b} + \\frac{c}{60-c} = 7\\). We can use this to help us find the value of the expression \\(5 \\left(\\frac{a}{25-a}\\right) + 13 \\left(\\frac{b}{65-b}\\right) + 12 \\left(\\frac{c}{60-c}\\right)\\).\n\nNotice that:\n\n\\[5 \\left(\\frac{a}{25-a}\\right) + 13 \\left(\\frac{b}{65-b}\\right) + 12 \\left(\\frac{c}{60-c}\\right) = 5 \\left(\\frac{1}{7-1} + \\frac{1}{65-63} + \\frac{1}{60-59}\\right) + 13 \\left(\\frac{1}{7-1} + \\frac{1}{65-63} + \\frac{1}{60-59}\\right).\\]\n\nSince \\(\\frac{a}{25-a} + \\frac{b}{65-b} + \\frac{c}{60-c} = 7\\), the expression simplifies to:\n\n\\[5 \\left(\\frac{1}{6} + \\frac{1}{2} + 1\\right) + 13 \\left(\\frac{1}{6} + \\frac{1}{2} + 1\\right) = 7 \\left(\\frac{1}{6} + \\frac{1}{2} + 1\\right) = 7 \\left(\\frac{1 + 3 + 6}{6}\\right) = 7 \\left(\\frac{10}{6}\\right) = 7 \\left(\\frac{5}{3}\\right) = \\frac{35}{3}.\\]\n\nThus, the value of the expression is:\n\n\\[\n\\boxed{\\frac{35}{3}}.\n\\]",
#            extraction_config=[LatexExtractionConfig()],
#        )
## Update the MDP tree with the new response
#mdp_tree.update(states, 1)
## Print the MDP tree to see the structure
#print(mdp_tree)
#exit()

# Read the json file train.json
with open("XR1-7500/extend/train.json", "r") as f:
    math_qa_dataset = json.load(f)

# Initialize an empty list to store the processed data
MDP_tree_math_qa_dataset = []
count = 0
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

    # Store the MDP tree 
    item["MDP_tree"] = problem_mdp_tree.__repr__()
    # Add the MDP tree to MDP_tree_math_qa_dataset
    MDP_tree_math_qa_dataset.append(item)
    count += 1
    if count == 5:
        break

# Save the processed data to a new json file
with open("train_MDP_tree.json", "w") as f:
    json.dump(MDP_tree_math_qa_dataset, f, indent=4)
    