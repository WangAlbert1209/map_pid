import os
import pickle

import matplotlib.pyplot as plt
import neat
import networkx as nx

from MAPELITE.map import Archive
from origin_neat.cart_pole import discrete_actuator_force, CartPole
from origin_neat.movie import make_movie


def visualize_neural_network(nodes, connections):
    """
    Visualizes a neural network based on nodes and connections.

    Parameters:
    - nodes: List of node dictionaries with keys like 'key', 'bias', etc.
    - connections: List of connection dictionaries with keys like 'key', 'weight', 'enabled'.
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
    layer_spacing = 2
    node_spacing = 1

    # Inputs on the left
    for i, node in enumerate(sorted(input_nodes)):
        pos[node] = (0, -i * node_spacing)

    # Hidden in the middle
    for i, node in enumerate(sorted(hidden_nodes)):
        pos[node] = (1, -i * node_spacing)

    # Outputs on the right
    for i, node in enumerate(sorted(output_nodes)):
        pos[node] = (2, -i * node_spacing)

    # Draw nodes
    plt.figure(figsize=(8, 6))

    # Draw input nodes
    nx.draw_networkx_nodes(G, pos,
                           nodelist=input_nodes,
                           node_shape='s',
                           node_color='lightblue',
                           node_size=1000,
                           label='Input')

    # Draw hidden nodes
    nx.draw_networkx_nodes(G, pos,
                           nodelist=hidden_nodes,
                           node_shape='o',
                           node_color='lightgreen',
                           node_size=1000,
                           label='Hidden')

    # Draw output nodes
    nx.draw_networkx_nodes(G, pos,
                           nodelist=output_nodes,
                           node_shape='s',
                           node_color='salmon',
                           node_size=1000,
                           label='Output')

    # Draw edges
    edge_weights = nx.get_edge_attributes(G, 'weight')
    rounded_weights = {edge: f"{weight:.2f}" for edge, weight in edge_weights.items()}
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, connectionstyle='arc3,rad=0.1')

    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=rounded_weights, font_color='red')

    # Draw labels
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, node_labels, font_size=12, font_color='black')

    # Create legend
    import matplotlib.patches as mpatches
    input_patch = mpatches.Patch(color='lightblue', label='Input')
    hidden_patch = mpatches.Patch(color='lightgreen', label='Hidden')
    output_patch = mpatches.Patch(color='salmon', label='Output')
    plt.legend(handles=[input_patch, hidden_patch, output_patch])

    plt.axis('off')
    plt.title('Neural Network Visualization')
    plt.show()


ll = 0
with open("./map_archive_pole.pkl", "rb") as f:
    archive = pickle.load(f)
    map = Archive(0, 0, is_cvt=True, cvt_file="../MAPELITE/centroids_1000_dim2.dat")
    map.archive = archive
    map.display_archive()
    # print(archive)
    # print(len(archive))
    for b, g in archive.items():
        print(g.fitness)
        if ll > 10:
            break
        if g.fitness > 59 and g.behavior[1]>0.5 and g.behavior[1]<0.6:
            ll += 1

            local_dir = os.path.dirname(__file__)
            config_path = os.path.join(local_dir, 'config_pole.ini')
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 config_path)

            net = neat.nn.FeedForwardNetwork.create(g, config)
            sim = CartPole()

            print()
            print("Initial conditions:")
            print("        x = {0:.4f}".format(sim.x))
            print("    x_dot = {0:.4f}".format(sim.dx))
            print("    theta = {0:.4f}".format(sim.theta))
            print("theta_dot = {0:.4f}".format(sim.dtheta))
            print()

            # Run the given simulation for up to 120 seconds.
            balance_time = 0.0
            while sim.t < 120.0:
                inputs = sim.get_scaled_state()
                action = net.activate(inputs)

                # Apply action to the simulated cart-pole
                force = discrete_actuator_force(action)
                sim.step(force)

                # Stop if the network fails to keep the cart within the position or angle limits.
                # The per-run fitness is the number of time steps the network can balance the pole
                # without exceeding these limits.
                if abs(sim.x) >= sim.position_limit or abs(sim.theta) >= sim.angle_limit_radians:
                    break

                balance_time = sim.t

            print('Pole balanced for {0:.1f} of 120.0 seconds'.format(balance_time))

            print()
            print("Final conditions:")
            print("        x = {0:.4f}".format(sim.x))
            print("    x_dot = {0:.4f}".format(sim.dx))
            print("    theta = {0:.4f}".format(sim.theta))
            print("theta_dot = {0:.4f}".format(sim.dtheta))
            print()
            print("Making movie...")
            make_movie(net, discrete_actuator_force, 15.0, f"./movie_pole/feedforward-movie_{ll}.mp4")
