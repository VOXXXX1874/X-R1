import graphviz
from collections import namedtuple

# Define a structure to hold the node data, matching the input format
number_node = namedtuple('number_node', ['index', 'existing_numbers', 'new_number', 'links', 'state_value'])

# This function simply returns its arguments, allowing us to evaluate the input string
def MDP_tree(*nodes):
    return nodes

def visualize_mdp_tree(data):
    """
    Generates a visualization of the MDP tree using Graphviz.
    """
    # Create a new directed graph
    dot = graphviz.Digraph('MDP_Tree', comment='MDP Tree Visualization')
    dot.attr(rankdir='TB', size='15,15', dpi='300')
    dot.attr('node', shape='record', style='rounded', fontname='Helvetica')
    dot.attr('edge', fontname='Helvetica', fontsize='10')

    # Add nodes to the graph
    for node in data:
        # Format the label for each node, avoiding problematic characters
        existing_str = str(sorted(list(node.existing_numbers)))
        label = (
            f"{{<f0> Index: {node.index} |"
            f"<f1> New Number: {node.new_number} |"
            f"<f2> Existing: {existing_str}}}"
        )
        dot.node(str(node.index), label)

    # Add edges to the graph
    for node in data:
        for link in node.links:
            dot.edge(str(node.index), str(link['node']), label=f"pass: {link['num_pass']}")

    # Render the graph to a file (e.g., PNG) and view it
    print("Rendering graph to mdp_tree.png...")
    dot.render('mdp_tree', format='png', view=True, cleanup=True)
    print("Done.")

if __name__ == '__main__':
    # Paste the data provided by the user
    tree_data = MDP_tree(number_node(index=0, existing_numbers=set(), new_number=None, links=[{'node': 1, 'num_pass': '21'}], state_value=0.29), number_node(index=1, existing_numbers=set(), new_number=2, links=[{'node': 2, 'num_pass': '8'}, {'node': 6, 'num_pass': '13'}], state_value=0.29), number_node(index=2, existing_numbers={2}, new_number=3, links=[{'node': 3, 'num_pass': '8'}], state_value=0.31), number_node(index=6, existing_numbers={2}, new_number=5, links=[{'node': 3, 'num_pass': '11'}, {'node': 18, 'num_pass': '2'}], state_value=0.27), number_node(index=3, existing_numbers={2, 3, 5}, new_number=5, links=[{'node': 4, 'num_pass': '10'}, {'node': 11, 'num_pass': '1'}, {'node': 15, 'num_pass': '6'}, {'node': 19, 'num_pass': '1'}, {'node': 25, 'num_pass': '1'}], state_value=0.31), number_node(index=18, existing_numbers={2, 5}, new_number=7, links=[{'node': 19, 'num_pass': '2'}], state_value=0.05), number_node(index=4, existing_numbers={2, 3, 5}, new_number=6, links=[{'node': 5, 'num_pass': '1'}, {'node': 7, 'num_pass': '7'}, {'node': 10, 'num_pass': '2'}], state_value=0.53), number_node(index=11, existing_numbers={2, 3, 5}, new_number=1, links=[{'node': 12, 'num_pass': '1'}], state_value=0.00), number_node(index=15, existing_numbers={2, 3, 5}, new_number=4, links=[{'node': 16, 'num_pass': '4'}, {'node': 23, 'num_pass': '2'}], state_value=0.10), number_node(index=19, existing_numbers={2, 3, 5, 7}, new_number=3, links=[{'node': 20, 'num_pass': '2'}, {'node': 16, 'num_pass': '1'}], state_value=0.05), number_node(index=25, existing_numbers={2, 3, 5}, new_number=0, links=[{'node': 26, 'num_pass': '1'}], state_value=0.00), number_node(index=5, existing_numbers=set(), new_number=None, links=[], state_value=1.00), number_node(index=7, existing_numbers={2, 3, 5, 6}, new_number=7, links=[{'node': 8, 'num_pass': '5'}, {'node': 14, 'num_pass': '1'}, {'node': 30, 'num_pass': '1'}], state_value=0.51), number_node(index=10, existing_numbers={2, 3, 5, 6}, new_number=4, links=[{'node': 8, 'num_pass': '1'}, {'node': 24, 'num_pass': '1'}], state_value=0.36), number_node(index=12, existing_numbers={1, 2, 3, 5}, new_number=7, links=[{'node': 13, 'num_pass': '1'}], state_value=0.00), number_node(index=16, existing_numbers={2, 3, 4, 5, 7}, new_number=7, links=[{'node': 17, 'num_pass': '3'}, {'node': 14, 'num_pass': '1'}, {'node': 8, 'num_pass': '1'}], state_value=0.14), number_node(index=23, existing_numbers={2, 3, 4, 5}, new_number=1, links=[{'node': 14, 'num_pass': '1'}, {'node': 29, 'num_pass': '1'}], state_value=0.00), number_node(index=20, existing_numbers={2, 3, 5, 7}, new_number=10, links=[{'node': 21, 'num_pass': '1'}, {'node': 14, 'num_pass': '1'}], state_value=0.00), number_node(index=26, existing_numbers={0, 2, 3, 5}, new_number=4, links=[{'node': 27, 'num_pass': '1'}], state_value=0.00), number_node(index=8, existing_numbers={2, 3, 4, 5, 6, 7}, new_number=4, links=[{'node': 9, 'num_pass': '5'}, {'node': 14, 'num_pass': '2'}], state_value=0.71), number_node(index=14, existing_numbers=set(), new_number=None, links=[], state_value=0.00), number_node(index=30, existing_numbers={2, 3, 5, 6, 7}, new_number=9, links=[{'node': 31, 'num_pass': '1'}], state_value=0.00), number_node(index=24, existing_numbers={2, 3, 4, 5, 6}, new_number=1, links=[{'node': 14, 'num_pass': '1'}], state_value=0.00), number_node(index=13, existing_numbers={1, 2, 3, 5, 7}, new_number=6, links=[{'node': 14, 'num_pass': '1'}], state_value=0.00), number_node(index=17, existing_numbers={2, 3, 4, 5, 7}, new_number=10, links=[{'node': 14, 'num_pass': '1'}, {'node': 22, 'num_pass': '2'}], state_value=0.00), number_node(index=29, existing_numbers={1, 2, 3, 4, 5}, new_number=7, links=[{'node': 14, 'num_pass': '1'}], state_value=0.00), number_node(index=21, existing_numbers={2, 3, 5, 7, 10}, new_number=0, links=[{'node': 14, 'num_pass': '1'}], state_value=0.00), number_node(index=27, existing_numbers={0, 2, 3, 4, 5}, new_number=7, links=[{'node': 28, 'num_pass': '1'}], state_value=0.00), number_node(index=9, existing_numbers={2, 3, 4, 5, 6, 7}, new_number=0, links=[{'node': 5, 'num_pass': '5'}], state_value=1.00), number_node(index=31, existing_numbers={2, 3, 5, 6, 7, 9}, new_number=4, links=[{'node': 32, 'num_pass': '1'}], state_value=0.00), number_node(index=22, existing_numbers={2, 3, 4, 5, 7, 10}, new_number=12, links=[{'node': 14, 'num_pass': '2'}], state_value=0.00), number_node(index=28, existing_numbers={0, 2, 3, 4, 5, 7}, new_number=1, links=[{'node': 14, 'num_pass': '1'}], state_value=0.00), number_node(index=32, existing_numbers={2, 3, 4, 5, 6, 7, 9}, new_number=10, links=[{'node': 14, 'num_pass': '1'}], state_value=0.00))

    visualize_mdp_tree(tree_data)