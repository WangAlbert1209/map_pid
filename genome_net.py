import math
import os
import pickle

import matplotlib.pyplot as plt
import neat
import networkx as nx
import sys
sys.path.append("C:\\Users\\77287\\Desktop\\map_pid")
from src.my_neat.myfeedfoward import MyFeedForwardNetwork
from src.my_neat.mygenome import CustomGenome
def visualize_neural_network(nodes, connections, show_weights=True):
    """
    Visualizes a neural network based on nodes and connections.

    Parameters:
    - nodes: List of node dictionaries with keys like 'key', 'bias', etc.
    - connections: List of connection dictionaries with keys like 'key', 'weight', 'enabled'.
    - show_weights: Boolean, whether to display connection weights (default: True)
    """
    G = nx.DiGraph()

    # Categorize nodes
    input_nodes = [-1, -2, -3, -4]
    output_nodes = []
    hidden_nodes = []

    for i, node in nodes.items():
        key = node.key
        if key < 0:
            input_nodes.append(key)
        elif key == 0:
            output_nodes.append(key)
        else:
            hidden_nodes.append(key)
        G.add_node(key, label=node.activation)
    G.add_node(-1, label="none")
    G.add_node(-2, label="none")
    G.add_node(-3, label="none")
    G.add_node(-4, label="none")

    # Add edges
    for i, conn in connections.items():
        if conn.enabled:
            from_node, to_node = conn.key
            weight = conn.weight
            G.add_edge(from_node, to_node, weight=weight)

    # Positioning nodes
    pos = {}
    layer_spacing = 3
    node_spacing = 1.5

    # Inputs on the left
    input_count = len(input_nodes)
    for i, node in enumerate(sorted(input_nodes)):
        y_pos = (input_count-1)/2 - i
        pos[node] = (0, y_pos * node_spacing)

    # Hidden in the middle
    hidden_count = len(hidden_nodes)
    for i, node in enumerate(sorted(hidden_nodes)):
        y_pos = (hidden_count-1)/2 - i
        pos[node] = (layer_spacing, y_pos * node_spacing)

    # Outputs on the right
    output_count = len(output_nodes)
    for i, node in enumerate(sorted(output_nodes)):
        y_pos = (output_count-1)/2 - i
        pos[node] = (layer_spacing * 2, y_pos * node_spacing)

    # Draw nodes
    plt.figure(figsize=(12, 8))

    # Draw input nodes
    nx.draw_networkx_nodes(G, pos,
                           nodelist=input_nodes,
                           node_shape='o',
                           node_color='lightblue',
                           node_size=1500,
                           edgecolors='darkblue',
                           linewidths=2,
                           label='Input')

    # Draw hidden nodes
    nx.draw_networkx_nodes(G, pos,
                           nodelist=hidden_nodes,
                           node_shape='o',
                           node_color='lightgreen',
                           node_size=1800,
                           edgecolors='darkgreen',
                           linewidths=2,
                           label='Hidden')

    # Draw output nodes
    nx.draw_networkx_nodes(G, pos,
                           nodelist=output_nodes,
                           node_shape='o',
                           node_color='salmon',
                           node_size=1800,
                           edgecolors='black',
                           linewidths=2,
                           label='Output')

    # Draw edges
    nx.draw_networkx_edges(G, pos,
                           edge_color='black',
                           width=2.5,
                           arrowsize=35,
                           arrowstyle='->',
                           arrows=True,
                           min_source_margin=20,
                           min_target_margin=20)

    # Draw edge labels (only if show_weights is True)
    if show_weights:
        edge_weights = nx.get_edge_attributes(G, 'weight')
        rounded_weights = {edge: f"{weight:.2f}" for edge, weight in edge_weights.items()}
        nx.draw_networkx_edge_labels(G, pos,
                                    edge_labels=rounded_weights,
                                    font_size=10,
                                    font_weight='bold',
                                    font_color='darkred',
                                    bbox=dict(alpha=0),
                                    label_pos=0.5,
                                    rotate=True)

    # Draw labels
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos,
                           node_labels,
                           font_size=10,
                           font_weight='bold',
                           font_color='black')

    # Create legend
    import matplotlib.patches as mpatches
    legend_elements = [
        mpatches.Patch(facecolor='lightblue', edgecolor='darkblue', label='Input'),
        mpatches.Patch(facecolor='lightgreen', edgecolor='darkgreen', label='Hidden'),
        mpatches.Patch(facecolor='salmon', edgecolor='darkred', label='Output')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)

    plt.title('Neural Network Visualization', fontsize=16, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('./best_genome.png')
    plt.show()


with open('./best_genome.pkl','rb') as f:
    g = pickle.load(f)

print(g)
# 显示权重
visualize_neural_network(g.nodes, g.connections, show_weights=False)
# 不显示权重
# visualize_neural_network(g.nodes, g.connections, show_weights=False)
