import unittest

import numpy as np

from CPPN.genome import Genome, NodeGene, EdgeGene
from CPPN.nn import NNGraph
from neat1 import NEAT
from utils.config import Config


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

    def test_config_new(self):
        config = Config.load('../config.json')
        neat = NEAT(config)
        g = neat.config_new()
        pops = neat.populate()
        print(neat.global_genome_id)
        print(neat.cur_nid)
        print(neat.cur_innovation)


if __name__ == '__main__':
    unittest.main()
