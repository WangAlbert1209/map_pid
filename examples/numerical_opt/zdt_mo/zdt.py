import numpy as np
from matplotlib import pyplot as plt

from MAPELITE.map import Archive
from utils.config import Config


def zdt1(x):
    n = len(x)
    f1 = x[0]
    g = 1 + 9 / (n - 1) * sum(x[1:-1], x[-1])
    h = 1 - np.sqrt(f1 / g)
    f2 = g * h
    return [-f1, -f2], [x[0], x[1]]


def zdt3(x):
    n = len(x)
    f1 = x[0]
    g = 1 + 9 / (n - 1) * sum(x[1:])
    h = 1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)
    f2 = g * h
    return [-f1, -f2], [x[0], x[1]]


def zdt5(x):
    n = len(x)
    f1 = x[0]
    g = 1 + 10 * (n - 1) + sum([(xi - 0.5) ** 2 for xi in x[1:]])
    h = 1 - np.sqrt(f1 / g)
    f2 = g * h
    return [-f1, -f2], [x[0], x[1]]


class Indv:
    def __init__(self, encode, fitness=None, behavior=None):
        self.encode = encode
        self.fitness = fitness
        self.behavior = behavior


def select_population(archive, population_size):
    # 获取所有的 cell 位置
    cell_positions = list(archive.archive.keys())
    # 选择 population_size 个 cell，通过索引选择
    cell_indices = np.random.choice(len(cell_positions), population_size, replace=True)

    selected_population = []

    for idx in cell_indices:
        # 获取当前 cell 中的个体（genomes），通过索引访问 cell
        cell_position = cell_positions[idx]
        cell_genomes = archive.archive[cell_position]

        # 从当前 cell 中随机选择一个个体
        selected_individual = np.random.choice(cell_genomes)
        # 添加到选中的个体列表
        selected_population.append(selected_individual)

    return selected_population


# 遗传算法实现
def binary_ga(archive, generations=500, population_size=200, chromosome_length=16, mutation_rate=0.1):
    def decode(chromosome):
        """将二进制染色体解码为浮点数（0~1）"""
        return int("".join(map(str, chromosome)), 2) / (2 ** (chromosome_length / 2) - 1)

    # 初始化种群
    population = [Indv(np.random.randint(2, size=chromosome_length)) for _ in range(population_size)]

    for generation in range(generations):
        # 计算适应度
        for indv in population:
            decoded = np.array(
                [decode(indv.encode[:chromosome_length // 2]), decode(indv.encode[chromosome_length // 2:])])
            fitness, behavior = zdt5(decoded)
            indv.fitness = fitness
            indv.behavior = behavior
            archive.add_to_archive(indv)  # 添加到存档

        fitness_values = np.array([indv.fitness for indv in population])
        selected_population = select_population(archive, population_size)

        # 交叉（单点交叉）
        new_population = []
        for i in range(0, population_size, 2):
            parent1, parent2 = selected_population[i], selected_population[(i + 1) % population_size]
            crossover_point = np.random.randint(1, chromosome_length - 1)
            child1_encode = np.concatenate([parent1.encode[:crossover_point], parent2.encode[crossover_point:]])
            child2_encode = np.concatenate([parent2.encode[:crossover_point], parent1.encode[crossover_point:]])
            new_population.append(Indv(child1_encode))
            new_population.append(Indv(child2_encode))

        # 变异
        for indv in new_population:
            mutation_mask = np.random.rand(chromosome_length) < mutation_rate
            indv.encode = np.logical_xor(indv.encode, mutation_mask).astype(int)

        population = new_population

        if generation % 50 == 0:
            archive.visualize_map()
            # plot_archive(archive)

    return archive


def plot_archive(archive):
    # 获取archive中所有个体的解码值（x, y）和对应的适应度（z）
    x_vals = []
    y_vals = []
    z_vals = []

    for indv in archive.archive.values():
        x_vals.append(indv.behavior[0] * 10 - 5)
        y_vals.append(indv.behavior[1] * 10 - 5)
        z_vals.append(indv.fitness)

    # 将x, y, z转为numpy数组
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    z_vals = np.array(z_vals)

    # 创建3D图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_vals, y_vals, z_vals, c=z_vals, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Fitness')
    ax.set_title('Fitness Landscape')

    plt.show()


config = Config.load('./config.json')
archive = Archive(config.archive_rows, config.archive_rows, is_cvt=config.is_cvt,
                  cvt_file=config.cvt_file, cell_size=config.cell_size)

optimized_archive = binary_ga(archive)
