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

def MDP_tree_from_string(
        str_representation: str,
):
    """
    Parses a string representation of an MDP_tree and returns an MDP_tree object.
    The string should be in the format returned by the __repr__ method of MDP_tree.
    """
    # Extract the content within MDP_tree(...)
    content_match = re.search(r'MDP_tree\((.*)\)', str_representation, re.DOTALL)
    if not content_match:
        return None
    content = content_match.group(1)

    # Regex to parse each number_node
    node_pattern = re.compile(r"number_node\(index=(?P<index>\d+), existing_numbers=(?P<existing_numbers>set\(\)|{.*?}), new_number=(?P<new_number>\w+), links=\[(?P<links>.*?)\], state_value=(?P<state_value>[-.\d]+)\)")
    
    nodes_data = []
    for match in node_pattern.finditer(content):
        nodes_data.append(match.groupdict())

    if not nodes_data:
        return None

    # Create all nodes first and store them in a dictionary indexed by their ID
    nodes = {}
    for data in nodes_data:
        index = int(data['index'])
        
        # Parse existing_numbers
        existing_numbers_str = data['existing_numbers']
        if existing_numbers_str == 'set()':
            existing_numbers = set()
        else:
            # From "{1, 2, 3}" to set({1, 2, 3})
            existing_numbers = set(map(int, existing_numbers_str.strip('{}').split(','))) if existing_numbers_str != '{}' else set()

        # Parse new_number
        new_number_str = data['new_number']
        new_number = int(new_number_str) if new_number_str != 'None' else None
        
        node = number_node(existing_numbers=existing_numbers, new_number=new_number, index=index)
        node.state_value = float(data['state_value'])
        nodes[index] = node

    # Now, establish the links
    for data in nodes_data:
        index = int(data['index'])
        current_node = nodes[index]
        
        links_str = data['links']
        if links_str:
            link_pattern = re.compile(r"{'node': (\d+), 'num_pass': '(\d+)'}")
            for link_match in link_pattern.finditer(links_str):
                target_node_index = int(link_match.group(1))
                num_pass = int(link_match.group(2))
                if target_node_index in nodes:
                    target_node = nodes[target_node_index]
                    current_node.links.append({'node': target_node, 'num_pass': num_pass})

    # Reconstruct the MDP_tree object
    # The root node is assumed to be the one with index 0
    if 0 not in nodes:
        return None # Or handle error appropriately
        
    root_node = nodes[0]
    
    # Create a dummy MDP_tree and then populate it
    # The constructor of MDP_tree requires expression_list, which we don't have.
    # So we create a dummy one and then replace its properties.
    mdp_tree = MDP_tree([], 0) 
    mdp_tree.root_node = root_node
    
    # Find correct and wrong states
    # A state is considered final if it has no outgoing links.
    # Correct state has value > 0.5, wrong state has value < 0.5
    for node in nodes.values():
        if not node.links:
            if node.state_value >= 0.5:
                mdp_tree.correct_state = node
            else:
                mdp_tree.wrong_state = node

    mdp_tree.node_count = len(nodes)
    
    return mdp_tree
        
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