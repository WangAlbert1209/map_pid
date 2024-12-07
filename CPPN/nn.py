import networkx as nx
import numpy as np


class NNGraph(object):
    def __init__(self, graph, input_id, output_id):
        self.graph = graph
        self.input_id = input_id
        self.output_id = output_id
        self._sorted_nodes = list(nx.topological_sort(self.graph))

    # TODO 拓扑排序的同级节点可以解耦
    def eval(self, inputs, dtype=np.float64):
        activations = {}
        # assign inputs
        for i, v in enumerate(inputs):
            activations[i] = v

        graph_edges = self.graph.edges
        graph_in_edges = self.graph.in_edges

        # hidden or output node
        for node in self._sorted_nodes:
            if node in activations:
                continue

            in_edges = graph_in_edges(node)

            # Calculate the weighted sum of incoming activations and add bias
            activation = np.sum(
                activations[in_node] * graph_edges[in_node, out_node]['conn'].weight
                for in_node, out_node in in_edges
            )
            node_dict = self.graph.nodes[node]
            # 加上bias
            activation += node_dict['node'].bias

            if node_dict['node'].f_activation is not None:
                activation = node_dict['node'].f_activation(activation)

            activations[node] = activation
        return np.array(
            [activations[node] for node in self.output_id],
            dtype=dtype
        )

    @classmethod
    def from_genome(cls, genome):
        return cls(genome.to_graph(), genome.input_nodes_id, genome.output_nodes_id)
