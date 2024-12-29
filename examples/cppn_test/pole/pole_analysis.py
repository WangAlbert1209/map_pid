import pickle
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from map_reader import visualize_neural_network

# 设置字体为 Arial
plt.rcParams['font.family'] = 'Arial'


def plot_activation_frequency(file_path, fitness_threshold=60, bar_width=0.25):
    """
    画出激活函数频率的柱状图

    参数:
    file_path: str, 归档文件路径
    fitness_threshold: int, 用于筛选 fitness 大于此值的个体
    bar_width: float, 柱子的宽度

    返回:
    None
    """
    # 读取数据
    nodes = []
    with open(file_path, 'rb') as f:
        archive = pickle.load(f)
        for k, g in archive.items():
            if g.fitness > fitness_threshold:
                node_names = []
                for n in g.nodes.values():
                    node_names.append(n.activation)  # 假设激活函数名存储在 `n.activation` 中
                nodes.append(node_names)

    # 打印 nodes 内容

    # 将所有激活函数名平铺成一个列表
    all_activations = [activation for individual in nodes for activation in individual]

    # 使用 Counter 统计每个激活函数出现的次数
    activation_counts = Counter(all_activations)

    # 获取激活函数名称和对应的计数
    activation_names = list(activation_counts.keys())
    activation_frequencies = list(activation_counts.values())

    # 设置 Seaborn 样式（去除网格）
    sns.set(style="white")  # 去除背景网格

    # 设置图形大小
    plt.figure(figsize=(12, 8))

    # 使用自定义颜色渐变
    colors = plt.cm.viridis(np.linspace(0, 1, len(activation_frequencies)))

    # 设置柱子的宽度
    bar_width = bar_width  # 可以调整柱子的宽度

    # 绘制柱状图，并设置柱子的边界颜色和宽度
    bars = plt.bar(activation_names, activation_frequencies, color=colors, width=bar_width, edgecolor='black',
                   linewidth=3)

    # 添加柱子的数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom', fontsize=16)

    # 设置标签和标题，指定字体为 Arial
    plt.xlabel('Activation Function', fontsize=14, fontname='Arial')
    plt.ylabel('Frequency', fontsize=14, fontname='Arial')

    # 旋转 x 轴标签，避免重叠
    plt.xticks(rotation=45, fontsize=12, fontname='Arial')

    # 关闭背景网格
    plt.grid(False)

    # 获取当前坐标轴
    ax = plt.gca()

    # 设置坐标轴颜色为黑色，并加粗
    ax.spines['top'].set_color('black')
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_color('black')
    ax.spines['right'].set_linewidth(2)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(2)

    # 设置坐标轴刻度线颜色和宽度
    ax.tick_params(axis='x', colors='black', width=2)
    ax.tick_params(axis='y', colors='black', width=2)

    # 显示图形
    plt.tight_layout()
    plt.show()


def count_node_and_edge_frequencies(file_path, fitness_threshold=60):
    """
    统计每个个体的节点数量和有效边数量，并计算不同数量的频率。

    参数:
    file_path: str, 归档文件路径
    fitness_threshold: int, 用于筛选 fitness 大于此值的个体

    返回:
    node_count_frequencies: dict, 每个节点数量对应的频率
    edge_count_frequencies: dict, 每个边数量对应的频率
    """
    node_count = []
    edge_count = []
    nums = 0
    with open(file_path, 'rb') as f:

        archive = pickle.load(f)
        for k, g in archive.items():
            if g.fitness >= fitness_threshold:
                print(g.fitness)
                nums += 1
                # if nums <= 10:
                #     visualize_neural_network(g.nodes, g.connections)
                num_nodes = int(len(g.nodes))  # 统计节点数量
                num_edges = len([cnn for cnn in g.connections.values() if cnn.enabled is True])  # 统计启用的边
                node_count.append(num_nodes)
                edge_count.append(num_edges)

    # 使用 Counter 统计每个节点个数和边个数的频率
    node_count_frequencies = Counter(node_count)
    edge_count_frequencies = Counter(edge_count)

    return node_count_frequencies, edge_count_frequencies


def plot_node_and_edge_frequency_distribution(file_path, fitness_threshold=60):
    """
    统计并绘制节点数量和边数量的频率分布图。

    参数:
    file_path: str, 归档文件路径
    fitness_threshold: int, 用于筛选 fitness 大于此值的个体
    """
    node_count_frequencies, edge_count_frequencies = count_node_and_edge_frequencies(file_path, fitness_threshold)

    # 获取节点个数和边个数及其对应的频率
    node_counts = list(node_count_frequencies.keys())
    node_frequencies = list(node_count_frequencies.values())

    edge_counts = list(edge_count_frequencies.keys())
    edge_frequencies = list(edge_count_frequencies.values())

    # 创建一个图形，分为两行一列
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 设置颜色映射
    node_colors = plt.cm.plasma(np.linspace(0, 1, len(node_counts)))
    edge_colors = plt.cm.plasma(np.linspace(0, 1, len(edge_counts)))

    # 绘制节点数量的频率分布柱状图
    bars_nodes = axes[0].bar(node_counts, node_frequencies, color=node_colors, edgecolor='black', width=0.3,
                             linewidth=3)
    axes[0].set_xlabel('Node Count', fontsize=16, fontname='Arial')
    axes[0].set_ylabel('Frequency', fontsize=16, fontname='Arial')
    axes[0].set_title('Distribution of Node Counts per Individual', fontsize=16, fontname='Arial')
    axes[0].tick_params(axis='x', colors='black', width=2)
    axes[0].tick_params(axis='y', colors='black', width=2)

    # 绘制边数量的频率分布柱状图
    bars_edges = axes[1].bar(edge_counts, edge_frequencies, color=edge_colors, edgecolor='black', width=0.5,
                             linewidth=3)
    axes[1].set_xlabel('Edge Count', fontsize=16, fontname='Arial')
    axes[1].set_ylabel('Frequency', fontsize=16, fontname='Arial')
    axes[1].set_title('Distribution of Edge Counts per Individual', fontsize=16, fontname='Arial')
    axes[1].tick_params(axis='x', colors='black', width=2)
    axes[1].tick_params(axis='y', colors='black', width=2)

    # 设置坐标轴颜色为黑色，并加粗
    for ax in axes:
        ax.spines['top'].set_color('black')
        ax.spines['top'].set_linewidth(2)
        ax.spines['right'].set_color('black')
        ax.spines['right'].set_linewidth(2)
        ax.spines['left'].set_color('black')
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_color('black')
        ax.spines['bottom'].set_linewidth(2)
        ax.grid(False)

    # 设置 x 轴范围，避免出现小数
    axes[0].set_xticks(node_counts)
    axes[1].set_xticks(edge_counts)

    # 显示图形
    plt.tight_layout()
    plt.show()


# 使用示例
file_path = './map_archive_pole.pkl'  # 归档文件路径
plot_activation_frequency(file_path)
plot_node_and_edge_frequency_distribution(file_path, fitness_threshold=60)
