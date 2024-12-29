import unittest

import numpy as np

from MY_CPPN.genome import Genome
from utils.config import Config


class TestGenome(unittest.TestCase):

    def setUp(self):
        # 初始化一个 Genome 对象
        self.config = Config.load('../config.json')
        self.genome = Genome(self.config)

    def test_mutate_activation(self):
        # 记录初始激活函数
        initial_activations = [node.f_activation for node in self.genome.nodes.values()]
        self.genome.draw()
        print(initial_activations)
        # 执行突变
        self.genome.mutate_activation()

        # 检查是否有节点的激活函数被突变
        post_mutations = [node.f_activation for node in self.genome.nodes.values()]
        self.genome.draw()
        print(post_mutations)
        self.assertNotEqual(initial_activations, post_mutations)

    def test_mutate_bias(self):
        # 记录初始偏置
        initial_biases = [node.bias for node in self.genome.nodes.values()]
        print(initial_biases)
        # 执行突变
        self.genome.mutate_bias()

        # 检查是否有节点的偏置被突变
        post_mutations = [node.bias for node in self.genome.nodes.values()]
        self.assertNotEqual(initial_biases, post_mutations)

    def test_mutate_weight(self):
        # 记录初始权重
        initial_weights = [conn.weight for conn in self.genome.enabled_conn]
        print(initial_weights)
        # 执行突变
        self.genome.mutate_weight()

        # 检查是否有边的权重被突变
        post_mutations = [conn.weight for conn in self.genome.enabled_conn]
        print(post_mutations)
        self.assertNotEqual(initial_weights, post_mutations)

    def test_mutate_add_conn(self):
        # 记录初始连接数
        initial_conn_count = len(self.genome.connections)

        # 执行变异添加连接
        self.genome.mutate_add_conn()

        # 检查连接数是否增加
        post_conn_count = len(self.genome.connections)
        self.assertGreaterEqual(post_conn_count, initial_conn_count)

    def test_mutate_split(self):
        # 记录初始连接和节点数
        initial_conn_count = len(self.genome.enabled_conn)
        initial_node_count = len(self.genome.nodes)
        print((initial_conn_count, initial_node_count))
        self.genome.draw()
        # 执行节点分裂
        for _ in range(100):
            self.genome.mutate_split(cur_node_id=10 + 1)

        self.genome.draw()
        # 检查节点数和连接数是否增加
        post_conn_count = len(self.genome.enabled_conn)
        post_node_count = len(self.genome.nodes)
        print((post_conn_count, post_node_count))
        self.assertGreater(post_conn_count, initial_conn_count)
        self.assertGreater(post_node_count, initial_node_count)

    def test_mutate_delete_conn(self):
        # 记录初始连接数
        initial_conn_count = len(self.genome.connections)
        self.genome.draw()
        # 执行变异删除连接
        self.genome.mutate_delete_conn()
        self.genome.draw()
        # 检查连接数是否减少
        post_conn_count = len(self.genome.connections)
        print(initial_conn_count, post_conn_count)
        self.assertLessEqual(post_conn_count, initial_conn_count)

    def test_mutate_delete_node(self):
        self.genome.mutate_split(cur_node_id=max(self.genome.node_ids) + 1)
        self.genome.mutate_split(cur_node_id=max(self.genome.node_ids) + 1)
        # 记录初始节点和连接数
        initial_conn_count = len(self.genome.connections)
        initial_node_count = len(self.genome.nodes)
        self.genome.draw()
        # 执行删除节点
        self.genome.mutate_delete_node()
        self.genome.draw()
        # 检查节点数和连接数是否减少
        post_conn_count = len(self.genome.connections)
        post_node_count = len(self.genome.nodes)
        self.assertLess(post_node_count, initial_node_count)
        self.assertLess(post_conn_count, initial_conn_count)

    def test_xover(self):
        # 创建两个基因组实例

        genome1 = Genome(self.config)
        genome2 = Genome(self.config)

        # 对每个基因组进行几次 mutate_split 操作
        sss = [9, 12, 10, 8]

        genome1.mutate_split(7 + 1)
        genome1.mutate_split(8 + 1)
        genome2.mutate_split(7 + 1)
        genome2.mutate_split(9 + 1)
        genome2.mutate_split(8 + 1)

        # 打印基因组的状态
        print("Genome 1 before crossover:")
        print(genome1)
        genome1.draw()
        print("Genome 2 before crossover:")
        print(genome2)
        genome2.draw()
        # 执行交叉操作
        child = genome1.xover(genome1, genome2)
        child.draw()
        # 打印子代基因组的状态
        print("Child genome after crossover:")
        print(child)



    def test_all(self):
        genome = Genome(self.config)
        genome.draw()
        # 添加
        genome.mutate_add_conn()
        genome.draw()
        genome.mutate_split(5)
        genome.draw()

        # 扰动
        genome.mutate_activation()
        genome.draw()
        genome.mutate_bias()
        genome.draw()
        genome.mutate_weight()
        genome.draw()

        # 删除
        genome.mutate_delete_conn()
        genome.draw()
        genome.mutate_delete_node()
        genome.draw()

if __name__ == '__main__':
    unittest.main()
