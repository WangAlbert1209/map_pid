import copy

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tabulate import tabulate

from MY_CPPN.nn import NNGraph
from utils.activations import gaussian, relu, sin, tanh, sigmoid, abs, square, log, exp

all_activations = {"None": None, 'gaussian': gaussian, 'relu': relu, 'sin': sin, 'tanh': tanh, 'sigmoid': sigmoid,
                   'abs': abs, 'square': square, 'log': log, 'exp': exp}


class NodeGene(object):
    def __init__(self, node_id, f_activation, gene_config, bias=None):
        self.node_id = node_id
        self.f_activation = f_activation
        self.gene_config = gene_config
        self.bias = bias if bias is not None else self.init_bias()

    def init_bias(self):
        # using gaussian distribution
        bias = np.random.normal(loc=self.gene_config.bias_mean, scale=self.gene_config.bias_std)
        return bias

    def perturb(self, sigma=0.1):
        self.bias += np.random.normal(0., sigma)

    def clone(self):
        return copy.deepcopy(self)


class EdgeGene(object):
    def __init__(self, a, b, gene_config, enabled=True, weight=None):
        self.innovation = (a, b)
        self.enabled = enabled
        self.a = a
        self.b = b
        self.gene_config = gene_config
        self.weight = weight if weight is not None else self.init_weight()

    def init_weight(self):
        # using gaussian distribution
        weight = np.random.normal(loc=self.gene_config.weight_mean, scale=self.gene_config.weight_std)
        return weight

    def perturb(self, sigma=0.1):
        self.weight += np.random.normal(0., sigma)

    def clone(self):
        # 不是引用
        return copy.deepcopy(self)

    @property
    def nodes(self):
        # 像属性使用方法 直接.nodes
        return self.a, self.b


class Genome(object):
    config = None

    def __init__(self, config):
        # {innovation（a,b）:edge}
        self.connections = {}
        # {nid(a):node}
        self.nodes = {}
        Genome.config = config
        self.input_num = Genome.config.num_inputs
        self.output_num = Genome.config.num_outputs
        self.hidden_num = Genome.config.num_hidden
        self.fitness = None
        self.behavior = None
        self.niche = None
        self.id = None
        self.activation_list = [all_activations[activation] for activation in self.config.activations]
        self.ancestor = None
        self.mutate_list = []
        self.mini_topology()

    # 添加连接只在这里进行
    def mini_topology(self):
        # 占据最小的数字
        for i in range(self.input_num):
            self.create_node(i)
        for i in range(self.output_num):
            # 输出函数的激活函数可以自己确定
            self.create_node(i + self.input_num, f_activation=all_activations[Genome.config.output_activation])

        for i in range(self.input_num):
            for j in range(self.output_num):
                self.create_connection(i, j + self.input_num)
        # TODO with hidden node initialize

    def create_node(self, node_id, f_activation=None):
        assert node_id not in self.nodes, f"Node with id {node_id} already exists."
        new_node = NodeGene(node_id=node_id, f_activation=f_activation, gene_config=self.config)
        self.nodes[node_id] = new_node
        return new_node

    def create_connection(self, in_id, out_id, weight=1.):
        assert (in_id, out_id) not in self.connections, f"conn with id {(in_id, out_id)} already exists."
        new_conn = EdgeGene(in_id, out_id, enabled=True, weight=weight, gene_config=self.config)
        self.connections[(in_id, out_id)] = new_conn
        return new_conn

    def mutate_activation(self):
        if len(self.hidden_nodes_id) == 0:
            return
        if np.random.rand() < Genome.config.mutation_activation_rate:
            mutate_node_id = np.random.choice(self.hidden_nodes_id)
            self.nodes[mutate_node_id].f_activation = np.random.choice(self.activation_list)

    def mutate_bias(self):
        mutate_node = None
        if np.random.rand() < Genome.config.mutation_rate:
            mutate_node = np.random.choice(list(self.nodes.values()))
            mutate_node.perturb(Genome.config.p_sigma)

    # TODO is there a threshold for the weight?
    def mutate_weight(self):
        mutate_conn = None
        if np.random.rand() < Genome.config.mutation_rate:
            mutate_conn = np.random.choice(self.enabled_conn)
            mutate_conn.perturb(Genome.config.p_sigma)

    def mutate_add_conn(self):
        def creates_cycle(connections, test):
            i, o = test
            if i == o:
                return True

            visited = {o}
            while True:
                num_added = 0
                for (a, b) in connections:
                    if a in visited and b not in visited:
                        if b == i:
                            return True

                        visited.add(b)
                        num_added += 1

                if num_added == 0:
                    return False

        if np.random.rand() > self.config.add_edge_rate:
            return False
        i, o = np.random.choice(self.node_ids, 2, False)
        # TODO redundant
        # 不能在out
        # if i in self.input_nodes_id and o in self.input_nodes_id or o in self.input_nodes_id:
        #     return False
        # # 不能都在in
        # if i in self.output_nodes_id and o in self.output_nodes_id or i in self.output_nodes_id:
        #     return False

        # 不合理
        if i in self.output_nodes_id or o in self.input_nodes_id:
            return False
        # 重复
        if (i, o) in self.edges:
            return False
        # 是否有环
        if creates_cycle(self.edges, (i, o)):
            return False
        # 创新号
        self.create_connection(i, o)
        return True

    # TODO 1. 会出现循环连接,原因是新id在已有id里出现 2. 删除连接就直接删除，disabled 会导致
    def mutate_split(self, cur_node_id):
        if np.random.rand() > self.config.split_edge_rate or len(self.enabled_conn) == 0:
            return False
        # 从所有enabled 连接中选一条 进行分裂，至少保证有一条的
        assert cur_node_id not in self.node_ids, f"split_node {cur_node_id}already exits!"
        selected_conn = np.random.choice(self.enabled_conn)
        s_a, s_b = selected_conn.a, selected_conn.b
        selected_conn.enabled = False
        del self.connections[selected_conn.innovation]
        new_node = self.create_node(cur_node_id, all_activations[self.config.default_activation])
        assert s_a != new_node.node_id
        self.create_connection(s_a, new_node.node_id)
        assert s_a != new_node.node_id
        self.create_connection(new_node.node_id, s_b)
        return True

    def mutate_delete_conn(self):
        # TODO 最小连通理论只是必要条件，非充分，特定情况（不能有环）会导致出错
        if np.random.rand() > self.config.delete_edge_rate or len(self.enabled_conn) <= len(self.nodes) - 1:
            return False

        selected_conn = np.random.choice(self.enabled_conn)
        selected_conn.enabled = False
        del self.connections[selected_conn.innovation]
        return True

    # TODO 删除导致的不连通？？
    def mutate_delete_node(self):
        # 选择一个隐藏节点进行删除
        if np.random.rand() > self.config.delete_node_rate or len(self.hidden_nodes_id) <= 1 or len(
                self.enabled_conn) <= len(self.nodes) - 1:
            return False
        selected_node_id = np.random.choice(self.hidden_nodes_id)
        # 删除关联边, 不能同时删除，访问时不能修改原数组
        connections_to_delete = []
        for (a, b), conn in list(self.connections.items()):
            if a == selected_node_id or b == selected_node_id:
                connections_to_delete.append((a, b))

        # 统一删除选定的连接
        for (a, b) in connections_to_delete:
            del self.connections[(a, b)]

        # 删除节点
        del self.nodes[selected_node_id]
        return True

    # 保证了父代完整，则xover后子代必定完整
    @classmethod
    # TODO： 同时融合亲本会导致有环,采用p1优先原则  2. edge应该是enable 的边来，否则出问题!
    def xover(cls, child, genome1, genome2):

        # Get nodes and edges from both parents
        nodes1, nodes2 = genome1.nodes, genome2.nodes
        edges1, edges2 = genome1.connections, genome2.connections

        # 复制边，不能同时复制节点，否则导致节点重复创建
        for (a, b) in edges1.keys():
            if (a, b) in edges1 and (a, b) in edges2:
                # Choose edge from either parent randomly
                if np.random.rand() < 0.5:
                    # 连接复制
                    child.connections[(a, b)] = edges1[(a, b)].clone()
                else:
                    child.connections[(a, b)] = edges2[(a, b)].clone()
            else:
                child.connections[(a, b)] = edges1[(a, b)].clone()

        # 复制节点
        for node_id in nodes1.keys():
            if node_id in nodes1 and node_id in nodes2:
                # Choose node from either parent randomly
                if np.random.rand() < 0.5:
                    child.nodes[node_id] = nodes1[node_id].clone()
                else:
                    child.nodes[node_id] = nodes2[node_id].clone()
            else:
                child.nodes[node_id] = nodes1[node_id].clone()

        return child

    @property
    def input_nodes_id(self):
        return [nid for nid in range(self.input_num)]

    @property
    def output_nodes_id(self):
        return [nid for nid in range(self.input_num, self.input_num + self.output_num)]

    # TODO 速度可以优化一下
    @property
    def hidden_nodes_id(self):
        return [nid for nid in self.nodes.keys() if nid not in self.input_nodes_id and nid not in self.output_nodes_id]

    @property
    def node_ids(self):
        return list(self.nodes.keys())

    # TODO 是实际删除还是做标记，要统一
    @property
    def enabled_conn(self):
        return [cnn for cnn in self.connections.values() if cnn.enabled]

    @property
    def edges(self):
        return [(cnn.a, cnn.b) for cnn in self.connections.values() if cnn.enabled]

    @property
    def get_dis_num(self):
        return len(self.connections) - len(self.enabled_conn)

    def to_graph(self):
        graph = nx.DiGraph()
        graph.add_nodes_from((node.node_id, dict(node=node)) for node in self.nodes.values())
        graph.add_edges_from(
            (conn.a, conn.b, dict(conn=conn)) for conn in self.enabled_conn
        )
        return graph

    def net_from_genome(self):
        return NNGraph(self.to_graph(), self.input_nodes_id, self.output_nodes_id)

    def __str__(self):
        # Nodes information
        node_table = []
        for node in self.nodes.values():
            activation_name = node.f_activation.__name__ if node.f_activation else 'none'
            node_table.append([node.node_id, activation_name])

        node_info = tabulate(node_table, headers=["Node ID", "Activation Function"], tablefmt="grid")

        # Edges information
        edge_table = []
        for conn in self.connections.values():
            edge_table.append([conn.a, conn.b, conn.weight])

        edge_info = tabulate(edge_table, headers=["From Node", "To Node", "Weight"], tablefmt="grid")

        # Summary information
        summary = f"Total Nodes: {len(self.nodes)} | Total Edges: {len(self.connections)}"

        return f"{node_info}\n\n{edge_info}\n\n{summary}"

    def draw(self):

        # 创建一个有向图
        G = nx.DiGraph()

        # 添加边并记录相关节点
        connected_nodes = set()
        for conn in self.enabled_conn:
            G.add_edge(conn.a, conn.b, weight=conn.weight)
            connected_nodes.add(conn.a)
            connected_nodes.add(conn.b)

        # 只保留与连接相关的节点
        connected_nodes = connected_nodes.union(set(self.input_nodes_id)).union(set(self.output_nodes_id))

        # 添加相关节点
        for node_id in connected_nodes:
            node = self.nodes[node_id]
            activation_name = node.f_activation.__name__ if node.f_activation else 'none'
            G.add_node(node_id, label=f'{node_id}\n{activation_name}')

        # 布局设置
        pos = {}
        input_nodes_id = set(self.input_nodes_id) & connected_nodes
        output_nodes_id = set(self.output_nodes_id) & connected_nodes
        hidden_nodes = connected_nodes - input_nodes_id - output_nodes_id

        # 设置输入节点的位置
        for i, node_id in enumerate(sorted(input_nodes_id)):
            pos[node_id] = (-1, i)

        # 设置输出节点的位置
        for i, node_id in enumerate(sorted(output_nodes_id)):
            pos[node_id] = (1, i)

        # 设置隐藏节点的位置
        for i, node_id in enumerate(sorted(hidden_nodes)):
            pos[node_id] = (0, i)

        # 绘制节点
        node_colors = []
        for node_id in G.nodes:
            if node_id in input_nodes_id:
                node_colors.append('green')
            elif node_id in output_nodes_id:
                node_colors.append('blue')
            else:
                node_colors.append('lightblue')

        node_labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_nodes(G, pos, node_size=100, node_color=node_colors)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_family='Arial')

        # 绘制边
        edge_labels = {(conn.a, conn.b): f'{conn.weight:.2f}' for conn in self.enabled_conn}
        nx.draw_networkx_edges(G, pos, edge_color='gray')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_family='Arial')

        # 显示图形
        plt.title("Genome Network Topology (Filtered for Connected Nodes)")
        plt.show()

