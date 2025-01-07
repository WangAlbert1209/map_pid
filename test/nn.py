import unittest
import numpy as np

from MY_CPPN.genome import Genome, NodeGene, EdgeGene
from MY_CPPN.nn import NNGraph


class TestNNGraph(unittest.TestCase):

    def setUp(self):
        # 创建一个简单的 Genome 对象，用于测试
        input_num = 3
        output_num = 2

        self.genome = Genome(input_num, output_num)

        # 创建节点
        for i in range(input_num + output_num):
            activation = np.tanh if i >= input_num else None
            self.genome.nodes[i] = NodeGene(i, activation)

        # 创建连接
        self.genome.connections[0] = EdgeGene(0, 0, 3, weight=0.5)
        self.genome.connections[1] = EdgeGene(1, 1, 3, weight=-0.7)
        self.genome.connections[2] = EdgeGene(2, 2, 4, weight=0.3)

        # 创建 NNGraph 对象
        self.nn_graph = NNGraph.from_genome(self.genome)

    def test_eval(self):
        # 定义输入向量
        inputs = np.array([1.0, 0.5, -1.5])

        # 计算输出
        output = self.nn_graph.eval(inputs)

        # 预期的输出（根据手动计算）
        node_3_activation = np.tanh(0.5 * 1.0 + (-0.7) * 0.5+0.5)
        node_4_activation = np.tanh(0.3 * (-1.5)+0.5)

        expected_output = np.array([node_3_activation, node_4_activation])

        # 比较实际输出和预期输出
        np.testing.assert_array_almost_equal(output, expected_output, decimal=10)

    def test_from_genome(self):
        # 检查从 genome 创建的图是否正确
        graph = self.nn_graph.graph

        # 验证节点数是否正确
        self.assertEqual(len(graph.nodes), 5)

        # 验证边数是否正确
        self.assertEqual(len(graph.edges), 3)

        # 验证边的权重是否正确
        for conn in self.genome.enabled_conn:
            self.assertEqual(graph[conn.a][conn.b]['conn'].weight, conn.weight)

if __name__ == '__main__':
    unittest.main()
