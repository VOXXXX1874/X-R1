# Import libraries to read json files and manipulate data
import json
# regular expression library for pattern matching
import re
from math_verify import LatexExtractionConfig
from math_verify.parser import *
import numpy as np
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

    def new_numbers_equal(self, another_node):
        """
        Compares this node with another number_node.
        Returns True if both nodes have the same existing and new numbers.
        """
        return self.new_number == another_node.new_number
    
    def existing_numbers_similarity(self, another_node):
        """
        Calculates the similarity between the existing numbers of this node and another number_node.
        Returns a float value representing the similarity.
        """
        common_numbers = self.existing_numbers.intersection(another_node.existing_numbers)
        total_numbers = self.existing_numbers.union(another_node.existing_numbers)
        return len(common_numbers) / len(total_numbers) if total_numbers else 0.0
    
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
    def __init__(self, response, final_reward):
        """
        Initializes the MDP_tree with a list of expressions and their end positions.
        Each expression is parsed into an expression_node.
        """
        self.root_node = number_node(existing_numbers=set(), new_number=None)
        self.node_list = [self.root_node]
        self.wrong_state = -1
        self.correct_state = -1
        self.update(response, final_reward)

    def __repr__(self):
        """
        Returns a string representation of the MDP_tree.
        """
        return f"MDP_tree with {len(self.node_list)} nodes: {[str(node) for node in self.node_list]}"
    
    def update(self, response, final_reward=0):
        """
        Updates the MDP_tree with new expressions and their end positions.
        This method can be used to add more nodes to the existing tree.
        """
        # Parse the numbers in the response
        numbers = response_parse(response)
        # Parse the new numbers and create nodes
        last_node_index = 0
        for i, number in enumerate(numbers):
            # if the new number is not new, continue
            if number in self.node_list[last_node_index].existing_numbers:
                continue
            # Create a new node with the existing numbers and the new number
            existing_numbers = deepcopy(self.node_list[last_node_index].existing_numbers)
            if self.node_list[last_node_index].new_number:
                existing_numbers.add(self.node_list[last_node_index].new_number)
            node = number_node(existing_numbers=existing_numbers, new_number=number)
            # Search for nodes that share the same new number
            similar_node_index = -1
            for i, existing_node in enumerate(self.node_list[1:], start=1):
                if existing_node.new_numbers_equal(node) and existing_node.existing_numbers_similarity(node) > 0.8:
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

            if i == len(numbers) - 1:
                node = number_node(existing_numbers=deepcopy(self.node_list[last_node_index].existing_numbers), new_number=None)
                if final_reward < 0.5:
                    if self.wrong_state == -1:
                        node.state_value = final_reward  # Set the last node's value to final_reward (final state)
                        self.node_list[last_node_index].update_link(len(self.node_list), 1)
                        self.wrong_state = len(self.node_list)
                        self.node_list.append(node)
                    else:
                        self.node_list[last_node_index].update_link(self.wrong_state, 1)
                        self.node_list[self.wrong_state].merge(node)
                else:
                    if self.correct_state == -1:
                        node.state_value = final_reward # Set the last node's value to final_reward (final state)
                        self.node_list[last_node_index].update_link(len(self.node_list), 1)
                        self.correct_state = len(self.node_list)
                        self.node_list.append(node)
                    else:
                        self.node_list[last_node_index].update_link(self.correct_state, 1)
                        self.node_list[self.correct_state].merge(node)


def response_parse(
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


# Test the MDP tree with a simple example
response1 = "To solve the given problem, we start with the equation:\n\n\\[\n\\frac{a}{25-a} + \\frac{b}{65-b} + \\frac{c}{60-c} = 7\n\\]\n\nWe need to evaluate the expression:\n\n\\[\n\\frac{5}{25-a} + \\frac{13}{65-b} + \\frac{12}{60-c}\n\\]\n\nFirst, let's make a substitution to simplify the problem. Let \\( x = 25 - a \\). Then, \\( a = 25 - x \\). Similarly, let \\( y = 65 - b \\), so \\( b = 65 - y \\), and let \\( z = 60 - c \\), so \\( c = 60 - z \\). Substituting these into the original equation, we get:\n\n\\[\n\\frac{a}{25-a} + \\frac{b}{65-b} + \\frac{c}{60-c} = \\frac{25-x}{x} + \\frac{65-y}{y} + \\frac{60-z}{z} = 7\n\\]\n\nThis can be rewritten as:\n\n\\[\n\\frac{25}{x} - 1 + \\frac{65}{y} - 1 + \\frac{60}{z} - 1 = 7\n\\]\n\nSimplifying the left side, we get:\n\n\\[\n\\frac{25}{x} + \\frac{65}{y} + \\frac{60}{z} - 3 = 7\n\\]\n\nAdding 3 to both sides, we obtain:\n\n\\[\n\\frac{25}{x} + \\frac{65}{y} + \\frac{60}{z} = 10\n\\]\n\nNow, we need to find the value of:\n\n\\[\n\\frac{5}{25-a} + \\frac{13}{65-b} + \\frac{12}{60-c}\n\\]\n\nSubstituting back \\( x = 25 - a \\), \\( y = 65 - b \\), and \\( z = 60 - c \\), we get:\n\n\\[\n\\frac{5}{x} + \\frac{13}{y} + \\frac{12}{z}\n\\]\n\nSince we already know that:\n\n\\[\n\\frac{25}{x} + \\frac{65}{y} + \\frac{60}{z} = 10\n\\]\n\nWe can conclude that:\n\n\\[\n\\frac{5}{x} + \\frac{13}{y} + \\frac{12}{z} = \\frac{1}{2} \\left( \\frac{25}{x} + \\frac{65}{y} + \\frac{60}{z} \\right) = \\frac{1}{2} \\times 10 = 5\n\\]\n\nTherefore, the value of the expression is:\n\n\\[\n\\boxed{5}\n\\]"
# Construct the MDP tree with the response
mdp_tree = MDP_tree(response1, 0)
# Update the MDP tree with new response
response2 = "To solve the given problem, we need to evaluate the expression \\(\\frac{5}{25-a} + \\frac{13}{65-b} + \\frac{12}{60-c}\\) given that \\(\\frac{a}{25-a} + \\frac{b}{65-b} + \\frac{c}{60-c} = 7\\). Let's denote \\(x = \\frac{a}{25-a}\\), \\(y = \\frac{b}{65-b}\\), and \\(z = \\frac{c}{60-c}\\). Then we have:\n\n\\[x + y + z = 7.\\]\n\nWe need to find the value of \\(5x + 13y + 12z\\). To do this, we can use the fact that \\(y = \\frac{1}{x^*}\\) and \\(z = \\frac{1}{x^* - 1}\\) where \\(x^*\\) is the value of the expression \\(\\frac{25-a}{a}\\) when \\(a = \\frac{1}{x}\\).\n\nFirst, let's express \\(y\\) and \\(z\\) in a different form. Notice that:\n\n\\[y = \\frac{1}{x^*} = \\frac{1}{\\frac{25-a}{a}} = \\frac{a}{25-a},\\]\n\\[z = \\frac{1}{x^* - 1} = \\frac{1}{\\frac{25-a}{a} - 1} = \\frac{1}{\\frac{25-a-a}{a}} = \\frac{a}{25-2a}.\\]\n\nNow, let's substitute \\(x = \\frac{a}{25-a}\\), \\(y = \\frac{b}{65-b}\\), and \\(z = \\frac{c}{60-c}\\) into the expression we need to evaluate:\n\n\\[5x + 13y + 12z = 5 \\left(\\frac{a}{25-a}\\right) + 13 \\left(\\frac{b}{65-b}\\right) + 12 \\left(\\frac{c}{60-c}\\right).\\]\n\nFrom the given condition, we know that \\(\\frac{a}{25-a} + \\frac{b}{65-b} + \\frac{c}{60-c} = 7\\). We can use this to help us find the value of the expression \\(5 \\left(\\frac{a}{25-a}\\right) + 13 \\left(\\frac{b}{65-b}\\right) + 12 \\left(\\frac{c}{60-c}\\right)\\).\n\nNotice that:\n\n\\[5 \\left(\\frac{a}{25-a}\\right) + 13 \\left(\\frac{b}{65-b}\\right) + 12 \\left(\\frac{c}{60-c}\\right) = 5 \\left(\\frac{1}{7-1} + \\frac{1}{65-63} + \\frac{1}{60-59}\\right) + 13 \\left(\\frac{1}{7-1} + \\frac{1}{65-63} + \\frac{1}{60-59}\\right).\\]\n\nSince \\(\\frac{a}{25-a} + \\frac{b}{65-b} + \\frac{c}{60-c} = 7\\), the expression simplifies to:\n\n\\[5 \\left(\\frac{1}{6} + \\frac{1}{2} + 1\\right) + 13 \\left(\\frac{1}{6} + \\frac{1}{2} + 1\\right) = 7 \\left(\\frac{1}{6} + \\frac{1}{2} + 1\\right) = 7 \\left(\\frac{1 + 3 + 6}{6}\\right) = 7 \\left(\\frac{10}{6}\\right) = 7 \\left(\\frac{5}{3}\\right) = \\frac{35}{3}.\\]\n\nThus, the value of the expression is:\n\n\\[\n\\boxed{\\frac{35}{3}}.\n\\]"
# Update the MDP tree with the new response
mdp_tree.update(response2, 0)
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
        
    