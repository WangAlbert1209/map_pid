import math

import numpy as np
from matplotlib import pyplot as plt

from MAPELITE.map import Archive
from utils.config import Config


def forward_kinematics(joint_positions, link_lengths):
    assert (len(joint_positions) == len(link_lengths))

    # some init
    p = np.append(joint_positions, 0)  # end-effector has no angle
    l = np.concatenate(([0], link_lengths))  # first link has no length
    joint_xy = np.zeros((len(p), 2))  # Cartesian positions of the joints
    mat = np.matrix(np.identity(4))  # 2D transformation matrix

    # compute the position of each joint
    for i in range(0, len(l)):
        m = [[math.cos(p[i]), -math.sin(p[i]), 0, l[i]],
             [math.sin(p[i]), math.cos(p[i]), 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]
        mat = mat * np.matrix(m)
        v = mat * np.matrix([0, 0, 0, 1]).transpose()
        joint_xy[i, :] = np.array(v[0:2].A.flatten())
    return joint_xy  # return the position of the joints


def arm_fit(genotype):
    # fitness is the standard deviation of joint angles (Smoothness)
    # (we want to minimize it)
    fit = 1 - np.std(genotype)

    # now compute the behavior
    #   scale to [0,2pi]
    g = np.interp(genotype, (0, 1), (0, 2 * math.pi))
    j = forward_kinematics(g, [1] * len(g))
    #  normalize behavior in [0,1]
    b = (j[-1, :]) / (2 * len(g)) + 0.5
    return fit, b


class Indv:
    def __init__(self, encode, fitness=None, behavior=None):
        self.encode = encode
        self.fitness = fitness
        self.behavior = behavior


# 遗传算法实现
def binary_ga(archive, generations=1000, population_size=200, chromosome_length=4, mutation_rate=0.1):
    # 初始化种群
    population = [Indv(np.random.rand(chromosome_length)) for _ in range(population_size)]

    for generation in range(generations):
        # 计算适应度
        for indv in population:
            fitness, behavior = arm_fit(indv.encode)
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
            indv.encode += np.random.normal(0, 0.1, chromosome_length) * mutation_mask
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
        x_vals.append(indv.behavior[0])
        y_vals.append(indv.behavior[1])
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
