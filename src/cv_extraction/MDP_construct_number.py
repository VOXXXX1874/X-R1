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
    def __init__(self, existing_numbers, new_number, index):
        self.index = index
        self.existing_numbers = existing_numbers
        self.new_number = new_number
        self.state_value = 0  # Placeholder for the value of the expression
        self.links = []  # Links to other nodes can be added later
        self.value_updated = False  # Flag to indicate if the value has been updated

    def __repr__(self):
        links_index = ", ".join(str({"node": link['node'].index, "num_pass": str(link['num_pass'])}) for link in self.links)
        # keep two decimal places for the state value
        return f"number_node(index={self.index}, existing_numbers={self.existing_numbers}, new_number={self.new_number}, links=[{links_index}], state_value={self.state_value:.2f})"
    
    def similarity(self, another_node):
        """
        Calculates the similarity between the existing numbers of this node and another number_node.
        Returns a float value representing the similarity.
        """
        this_node_number_set = self.existing_numbers.union({self.new_number}) if self.new_number != None else self.existing_numbers
        another_node_number_set = another_node.existing_numbers.union({another_node.new_number}) if another_node.new_number != None else another_node.existing_numbers
        # Calculate the similarity based on the existing numbers
        common_numbers = this_node_number_set.intersection(another_node_number_set)
        total_numbers = this_node_number_set.union(another_node_number_set)
        similarity_score = len(common_numbers) / len(total_numbers) if total_numbers else 0
        return similarity_score
    
    def update_link(self, node, num_pass):
        """
        Updates the link to another node based on the node index and the number of passes.
        This method can be used to establish connections between nodes in the tree.
        """
        for link in self.links:
            if link['node'].index == node.index:
                # If the link already exists, update the number of passes
                link['num_pass'] += num_pass
                return
        else:
            self.links.append({'node': node, 'num_pass': num_pass})

    def sync_links(self):
        """
        If there are multiple links to the same node, sum them up.
        """
        link_dict = {}
        for link in self.links:
            if link['node'].index in link_dict:
                link_dict[link['node'].index]['num_pass'] += link['num_pass']
            else:
                link_dict[link['node'].index] = {'node': link['node'], 'num_pass': link['num_pass']}
        self.links = list(link_dict.values())

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
        self.root_node = number_node(existing_numbers=set(), new_number=None, index=0)
        self.wrong_state = None
        self.correct_state = None
        self.node_count = 1
        self.update(expression_list, final_reward)

    def __repr__(self):
        """Returns a string representation of the MDP_tree through bfs traversal."""
        nodes = []
        queue = [self.root_node]
        visited = set()
        while queue:
            current_node = queue.pop(0)
            if current_node.index in visited:
                continue
            visited.add(current_node.index)
            nodes.append(str(current_node))
            for link in current_node.links:
                queue.append(link['node'])
        return "MDP_tree(" + ", ".join(nodes) + ")"
    
    def __len__(self):
        """Count the real number of nodes in the MDP_tree through bfs traversal."""
        count = 0
        queue = [self.root_node]
        visited = set()
        while queue:
            current_node = queue.pop(0)
            if current_node.index in visited:
                continue
            visited.add(current_node.index)
            count += 1
            for link in current_node.links:
                queue.append(link['node'])
        return count

    def bfs_similarity_search(self, node, similarity_threshold=1.0):
        """
        Performs a breadth-first search to find a node that is similar to the given node.
        Returns the similar node and its index if found, otherwise returns None and -1.
        """
        queue = [self.root_node]
        visited = set()
        while queue:
            current_node = queue.pop(0)
            if current_node.index in visited:
                continue
            visited.add(current_node.index)
            if current_node.similarity(node) >= similarity_threshold:
                return current_node, current_node.index
            for link in current_node.links:
                queue.append(link['node'])
        # If no similar node is found, return None and -1
        return None, -1
    
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
        last_node = self.root_node
        for i, number in enumerate(numbers):
            # if the new number is not new, continue
            if number in last_node.existing_numbers or number == last_node.new_number:
                continue
            # Create a new node with the existing numbers and the new number
            existing_numbers = deepcopy(last_node.existing_numbers)
            if last_node.new_number != None:
                existing_numbers.add(last_node.new_number)
            node = number_node(existing_numbers=existing_numbers, new_number=number, index=self.node_count)
            # Search for nodes that share the same new number
            similar_node, similar_node_index = self.bfs_similarity_search(
                node,
                similarity_threshold=1.0,
            )
            # If a similar node is found, merge it; otherwise, add a new node
            if similar_node_index != -1:
                last_node.update_link(similar_node, 1)
                last_node = similar_node
                similar_node.merge(node)
            else:
                last_node.update_link(node, 1)
                last_node = node
                self.node_count += 1

        if final_reward < 0.5:
            if self.wrong_state == None:
                node = number_node(existing_numbers=set(), new_number=None, index=self.node_count)
                self.node_count += 1
                node.state_value = final_reward  # Set the last node's value to final_reward (final state)
                last_node.update_link(node, 1)
                self.wrong_state = node
            else:
                last_node.update_link(self.wrong_state, 1)
        else:
            if self.correct_state == None:
                node = number_node(existing_numbers=set(), new_number=None, index=self.node_count)
                self.node_count += 1
                node.state_value = final_reward # Set the last node's value to final_reward (final state)
                last_node.update_link(node, 1)
                self.correct_state = node
            else:
                last_node.update_link(self.correct_state, 1)

    def trim(self):
        """
        Trims the MDP_tree by removing branch that directly leads to the wrong state by dfs
        """
        def dfs(node):
            # Perform a depth-first search to find and remove branches leading to the wrong state
            for link in node.links:
                if dfs(link['node']):
                    link['node'] = self.wrong_state
            node.sync_links()  # Sync links to remove duplicates
            if len(node.links) == 1 and node.links[0]["node"].index == self.wrong_state.index:
                # If the node has only one link to the wrong state, remove it
                return True
            # If the node has no links or more than one link, keep it
            return False

        # Start DFS from the root node
        dfs(self.root_node)

    def update_node_value(self):
        """
        Recursively updates the value of each node base on the value of the final state.
        The value of each node is weighted sum of the values of its children.
        """
        def dfs(node):
            if node.index == self.wrong_state.index or node.index == self.correct_state.index or node.value_updated:
                # If the node is a final state or already updated, return its value
                return node.state_value
            total_value = 0
            total_links = 0
            # Recursively calculate the value of each child node
            # and update the total value and total links
            for link in node.links:
                child_value = dfs(link['node'])
                total_value += child_value * link['num_pass']
                total_links += link['num_pass']
            if total_links > 0:
                node.state_value = total_value / total_links
            else:
                node.state_value = 0
            node.value_updated = True  # Mark the node as updated
            return node.state_value
        
        def reset_value_updated_flag_bfs(node):
            """Resets the value_updated flag for all nodes in the tree using BFS."""
            queue = [node]
            visited = set()
            while queue:
                current_node = queue.pop(0)
                if current_node.index in visited:
                    continue
                visited.add(current_node.index)
                current_node.value_updated = False
                for link in current_node.links:
                    queue.append(link['node'])

        # Start DFS from the root node
        dfs(self.root_node)
        # Reset the "value_updated" flag for all nodes
        reset_value_updated_flag_bfs(self.root_node)
        
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
#mdp_tree.update(states, 0)
## Print the MDP tree to see the structure
#mdp_tree.trim()  # Optional: trim the tree to remove branches leading to wrong state
#print(mdp_tree)
#exit()

# Read the json file train.json
with open("XR1-hard/extend20/train.json", "r") as f:
    math_qa_dataset = json.load(f)

# Initialize an empty list to store the processed data
MDP_tree_math_qa_dataset = []
count = 0
# Iterate through each item in the dataset
for item in math_qa_dataset:
    # Extract the solution, correct_responses, and wrong_responses
    solution = item.get("solution", "")
    correct_responses = item.pop("correct_responses", [])
    wrong_responses = item.pop("wrong_responses", [])
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
    # Trim the MDP tree if it is too large
    if len(problem_mdp_tree) > 100:
        problem_mdp_tree.trim()
    # Update the node values in the MDP tree
    problem_mdp_tree.update_node_value()
    # Store the MDP tree 
    item["MDP_tree"] = problem_mdp_tree.__repr__()
    # Add the MDP tree to MDP_tree_math_qa_dataset
    MDP_tree_math_qa_dataset.append(item)
    count += 1
    print(f"Processed {count} items.")

# Save the processed data to a new json file
with open("train_MDP_tree.json", "w") as f:
    json.dump(MDP_tree_math_qa_dataset, f, indent=4)
    