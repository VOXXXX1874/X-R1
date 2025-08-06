# Import libraries to read json files and manipulate data
import json
# regular expression library for pattern matching
import re
from math_verify import LatexExtractionConfig
from math_verify.parser import *
from copy import deepcopy
from ast import literal_eval

class number_node:
    """
    A class to represent a state as sequence of numbers
    The node contains existing numbers and new number
    It also contains value of this state and links to other states.
    """
    def __init__(self, existing_numbers, new_number, index):
        self.index = index
        self.existing_numbers = existing_numbers
        if new_number is not None:
            self.existing_numbers.add(new_number)
        self.state_value = 0  # Placeholder for the value of the expression
        self.links = []  # Links to other nodes can be added later
        self.value_updated = False  # Flag to indicate if the value has been updated

    def __repr__(self):
        links_index = ", ".join(str({"node": link['node'].index, "num_pass": str(link['num_pass'])}) for link in self.links)
        # keep two decimal places for the state value
        return f"number_node(index={self.index}, existing_numbers={self.existing_numbers}, links=[{links_index}], state_value={self.state_value:.2f})"
    
    def similarity(self, another_node):
        """
        Calculates the similarity between the existing numbers of this node and another number_node.
        Returns a float value representing the similarity.
        """
        # Calculate the similarity based on the existing numbers
        common_numbers = self.existing_numbers.intersection(another_node.existing_numbers)
        total_numbers = self.existing_numbers.union(another_node.existing_numbers)
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
        This method combines the numbers from both nodes.
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
            if number in last_node.existing_numbers:
                continue
            # Create a new node with the existing numbers and the new number
            existing_numbers = deepcopy(last_node.existing_numbers)
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

    def advantages_and_probability(self, expression_list, positions, generation_length, final_reward=0):
        """
        Calculate the advantage of each action (represented by expressions) based on the MDP tree.
        Return a list of advantages that contain the advantage for each token in the generation.
        At the same time, it also calculates the probability of the whole generation.
        """
        # Parse the numbers in the expression_list
        numbers_positions = []
        for expression, position in zip(expression_list, positions):
            # Extract numbers from the expression using regex
            numbers = number_parse(expression)
            for number in numbers:
                numbers_positions.append((number, position))
        # Parse the new numbers and match nodes
        advantages = []
        last_node = self.root_node
        last_position = 0
        probability = 1.0  # Initialize the probability of the generation
        for number, position in numbers_positions:
            # if the new number is not new, continue
            if number in last_node.existing_numbers:
                continue
            # if the new number is new, create a numbers set and find the matching node
            existing_numbers = deepcopy(last_node.existing_numbers)
            set_to_match = existing_numbers.union({number})
            # Search for nodes that share the same new number through the links
            for link in last_node.links:
                if link['node'].existing_numbers == set_to_match:
                    # If a matching node is found, calculate the advantage
                    advantage = link['node'].state_value - last_node.state_value
                    advantages.extend([advantage] * (position - last_position))
                    probability *= link['num_pass'] / sum(_link['num_pass'] for _link in last_node.links)
                    last_node = link['node']
                    last_position = position
                    break
            else:
                # There should be no case where a new number is not found in the links
                print(f"Warning: New number set {existing_numbers} not found in the links of node {last_node}.")

        # Finally, we deal with the terminal state
        if final_reward < 0.5:
            advantages.extend([self.wrong_state.state_value - last_node.state_value] * (generation_length - last_position))
            for link in last_node.links:
                if link['node'].index == self.wrong_state.index:
                    probability *= link['num_pass'] / sum(_link['num_pass'] for _link in last_node.links)
                    break
        else:
            advantages.extend([self.correct_state.state_value - last_node.state_value] * (generation_length - last_position))
            for link in last_node.links:
                if link['node'].index == self.correct_state.index:
                    probability *= link['num_pass'] / sum(_link['num_pass'] for _link in last_node.links)
                    break

        # Ensure the length of advantages matches the generation length
        if len(advantages) < generation_length:
            advantages.extend([0] * (generation_length - len(advantages)))
        elif len(advantages) > generation_length:
            advantages = advantages[:generation_length]
        return advantages, probability

    def bfs_decay(self, decay_rate=0.99):
        """
        Decays the number of passes in the MDP tree.
        This method can be used to reduce the influence of older actions over time.
        """
        queue = [self.root_node]
        visited = set()
        while queue:
            current_node = queue.pop(0)
            if current_node.index in visited:
                continue
            visited.add(current_node.index)
            # Decay the number of passes for each link
            for link in current_node.links:
                link['num_pass'] *= decay_rate
                queue.append(link['node'])
            current_node.sync_links()
        
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

def MDP_tree_from_string(
        str_representation: str,
):
    """
    Parses a string representation of an MDP_tree and returns an MDP_tree object.
    The string should be in the format returned by the __repr__ method of MDP_tree.
    """
    if not str_representation or not str_representation.startswith("MDP_tree("):
        return None

    # Extract the content inside MDP_tree(...)
    content = str_representation[len("MDP_tree("):-1]
    if not content:
        return MDP_tree([], 0)

    # Regex to parse each number_node
    node_pattern = re.compile(r"number_node\(index=(?P<index>\d+), existing_numbers=(?P<numbers>set\(\)|{.*?}), links=\[(?P<links>.*?)\], state_value=(?P<value>[\d\.-]+)\)")
    
    nodes_data = {}
    for match in node_pattern.finditer(content):
        data = match.groupdict()
        index = int(data['index'])
        
        # Safely evaluate existing_numbers
        try:
            existing_numbers = literal_eval(data['numbers'])
        except (ValueError, SyntaxError):
            existing_numbers = set()

        state_value = float(data['value'])
        
        # Parse links
        links_str = data['links']
        links = []
        if links_str:
            link_pattern = re.compile(r"{'node': (\d+), 'num_pass': '(\d+)'}")
            for link_match in link_pattern.finditer(links_str):
                links.append({'node': int(link_match.group(1)), 'num_pass': int(link_match.group(2))})

        nodes_data[index] = {
            'existing_numbers': existing_numbers,
            'state_value': state_value,
            'links': links
        }

    if not nodes_data:
        return MDP_tree([], 0)

    # Create node objects
    node_objects = {}
    for index, data in nodes_data.items():
        node = number_node(existing_numbers=data['existing_numbers'], new_number=None, index=index)
        node.state_value = data['state_value']
        node_objects[index] = node

    # Link nodes
    for index, data in nodes_data.items():
        current_node = node_objects[index]
        for link_info in data['links']:
            target_node_index = link_info['node']
            if target_node_index in node_objects:
                target_node = node_objects[target_node_index]
                current_node.update_link(target_node, link_info['num_pass'])

    # Create the MDP_tree object
    # Assuming the root is always index 0, and final_reward is not stored in the string repr
    # We can create a dummy MDP_tree and then set its root.
    mdp_tree = MDP_tree([], 0) 
    if 0 in node_objects:
        mdp_tree.root_node = node_objects[0]
    
    # Find correct and wrong states
    max_node_index = -1
    if node_objects:
        max_node_index = max(node_objects.keys())

    for node in node_objects.values():
        if not node.links: # Final states have no outgoing links
            if node.state_value >= 0.5:
                mdp_tree.correct_state = node
            else:
                mdp_tree.wrong_state = node

    mdp_tree.node_count = max_node_index + 1
    
    return mdp_tree

test_case = 1

# Test the MDP_tree_from_string function
if test_case == 0:
    # Read the json file train.json
    with open("src/cv_extraction/XR1-hard/MDP_extend_20/train.json", "r") as f:
        math_qa_dataset = json.load(f)

    for item in math_qa_dataset:
        # Get the 'MDP_tree' field from the item
        mdp_tree_str = item.get('MDP_tree', None)
        # reconstruct the MDP_tree from the string representation
        mdp_tree = MDP_tree_from_string(mdp_tree_str)
        another_mdp_tree_str = mdp_tree.__repr__()
        # Check if the reconstruction is equal to the original string
        if len(another_mdp_tree_str) != len(mdp_tree_str):
            print(f"Reconstruction failed for item with id {item.get('id', 'unknown')}.")
            print(f"1: {mdp_tree_str}")
            print(f"2: {another_mdp_tree_str}")

elif test_case == 1: 
    # Read the json file train.json
    with open("src/cv_extraction/XR1-hard/extend20/train.json", "r") as f:
        math_qa_dataset = json.load(f)

    item = math_qa_dataset[0]
    # Extract the solution, correct_responses, and wrong_responses
    solution = item.get("solution", "")
    correct_responses = item.pop("correct_responses", [])
    wrong_responses = item.pop("wrong_responses", [])
    generation = correct_responses[0]
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

    # Calculate the advantages and probability for the generation
    generation_states, generation_positions = thinking_parse(
        generation,
        extraction_config=[LatexExtractionConfig()],
    )
    problem_mdp_tree.update_node_value()
    advantages, probability = problem_mdp_tree.advantages_and_probability(
        generation_states,
        generation_positions,
        generation_length=len(generation),
        final_reward=1,
    )
    print("Question: ", item['problem'])
    print("Generation: ", generation)
    last_advantage_pos = -1
    for i, advantage in enumerate(advantages):
        if advantage != advantages[last_advantage_pos]:
            print(f"In the {last_advantage_pos} to {i} position, the advantage is {advantages[last_advantage_pos]}.")
            print(f"In that position, the generation is: {generation[last_advantage_pos + 1:i]}")
            print("---" * 10)
            last_advantage_pos = i
    if last_advantage_pos + 1 < len(generation):
        print(f"In the {last_advantage_pos + 1} to {len(generation)} position, the advantage is {advantages[last_advantage_pos]}.")
        print(f"In that position, the generation is: {generation[last_advantage_pos + 1:]}")
        print("---" * 10)
    print(f"Probability: {probability}")
