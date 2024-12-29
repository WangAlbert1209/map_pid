import os
import sys
import neat
import neat.genome
from neat.genes import DefaultConnectionGene, DefaultNodeGene
sys.path.append("C:\\Users\\77287\\Desktop\\map_pid")  # 将项目根目录添加到 Python 路径

from src.my_neat.pid_act import CustomActivationFunctionSet
class CustomGenomeConfig(neat.genome.DefaultGenomeConfig):
    def __init__(self, param_dict):
        super().__init__(param_dict)
        self.activation_defs = CustomActivationFunctionSet()

class CustomGenome(neat.DefaultGenome):
    """
    A custom genome class that extends DefaultGenome with overridden methods.
    支持类形式的激活函数自动实例化。
    """
    def __init__(self, key):
        super().__init__(key)
        
    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = DefaultNodeGene
        param_dict['connection_gene_type'] = DefaultConnectionGene
        return CustomGenomeConfig(param_dict)

    @staticmethod
    def create_node(config, node_id):
        """重写创建节点的方法，对需要实例化的激活函数进行处理"""
        node = config.node_gene_type(node_id)
        node.init_attributes(config)
        
        # 获取激活函数定义
        if hasattr(node, 'activation'):
            # print(config.activation_defs.functions)
            activation_type = config.activation_defs.get(node.activation)
            # 检查是否是类而不是函数
            if isinstance(activation_type, type):
                node.activation_function = activation_type()
        return node

    def mutate(self, config):
        """重写变异方法，确保激活函数正确实例化"""
        super().mutate(config)
        
        for node in self.nodes.values():
            if hasattr(node, 'activation'):
                activation_type = config.activation_defs.get(node.activation)
                # 如果是类类型且尚未实例化，则进行实例化
                if (isinstance(activation_type, type) and 
                    not isinstance(getattr(node, 'activation', None), activation_type)):
                    node.activation_function = activation_type()


    def reset_pid(self):
        """重置所有具有reset方法的激活函数的状态"""
        for node in self.nodes.values():
            if hasattr(node.activation_function, 'reset'):
                node.activation_function.reset()
