from neat.config import Config, ConfigParameter



class MyConfig(Config):
   # 定义新的参数列表，包含父类的所有参数和新参数
    __params = Config._Config__params + [
        ConfigParameter('gen', int),
        ConfigParameter('cvt_path', str),
        # 可以添加更多参数...
    ]

    def __init__(self, genome_type, reproduction_type, species_set_type, stagnation_type, filename):
        # 临时替换参数列表为扩展后的版本
        Config._Config__params = self.__params
        
        # 调用父类的初始化方法
        super().__init__(genome_type, reproduction_type, species_set_type, stagnation_type, filename)