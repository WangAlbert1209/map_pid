import math

import numpy as np
from matplotlib import pyplot as plt

from MAPELITE.map import Archive
from utils.config import Config


def rastrigin(xx):
    x = xx * 10 - 5  # scaling to [-5, 5]
    f = 10 * x.shape[0] + (x * x - 10 * np.cos(2 * math.pi * x)).sum()
    return f, np.array([xx[0], xx[1]])


class Indv:
    def __init__(self, encode, fitness=None, behavior=None):
        self.encode = encode
        self.fitness = fitness
        self.behavior = behavior


# 遗传算法实现
def binary_ga(archive, generations=500, population_size=5000, chromosome_length=16, mutation_rate=0.1):
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
            fitness, behavior = rastrigin(decoded)
            indv.fitness = fitness
            indv.behavior = behavior
            archive.add_to_archive(indv)  # 添加到存档
        # 输出当前最佳解
        best_in_archive = max(archive.archive.values(), key=lambda x: x.fitness)
        print(
            f"Best in Archive at Generation {generation + 1}: Fitness = {best_in_archive.fitness:.5f}, Behavior = {best_in_archive.behavior}")

        fitness_values = np.array([indv.fitness for indv in population])
        selected_population = np.random.choice(list(archive.archive.values()), population_size, True)

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
            archive.display_archive()
            plot_archive(archive)

    return archive


def plot_archive(archive):
    # 获取archive中所有个体的解码值（x, y）和对应的适应度（z）
    x_vals = []
    y_vals = []
    z_vals = []

    for indv in archive.archive.values():
        x_vals.append(indv.behavior[0]*10-5)
        y_vals.append(indv.behavior[1]*10-5)
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
                  cvt_file=config.cvt_file)

optimized_archive = binary_ga(archive)
