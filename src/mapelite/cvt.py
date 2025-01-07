from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import KMeans


def centroids_filename(k, dim):
    return 'centroids_' + str(k) + '_dim' + str(dim) + '.dat'


def write_centroids(centroids):
    k = centroids.shape[0]
    dim = centroids.shape[1]
    filename = centroids_filename(k, dim)
    with open(filename, 'w') as f:
        for p in centroids:
            for item in p:
                f.write(str(item) + ' ')
            f.write('\n')


def make_hashable(array):
    return tuple(map(float, array))


def cvt(k, dim, samples, cvt_use_cache=True):
    # check if we have cached values
    fname = centroids_filename(k, dim)
    if cvt_use_cache:
        if Path(fname).is_file():
            print("WARNING: using cached CVT:", fname)
            return np.loadtxt(fname)
    # otherwise, compute cvt
    print("Computing CVT (this can take a while...):", fname)

    x = np.random.rand(samples, dim)
    k_means = KMeans(init='k-means++', n_clusters=k,
                     n_init=1, verbose=1)  # ,algorithm="full")
    k_means.fit(x)
    write_centroids(k_means.cluster_centers_)
    return k_means.cluster_centers_

def plot_voronoi(centroids, bounds):
    # 使用 Voronoi 生成区域
    vor = Voronoi(centroids)

    # 绘制 Voronoi 图
    fig, ax = plt.subplots(figsize=(8, 8))
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6, point_size=2)

    # 绘制质心位置
    ax.scatter(centroids[:, 0], centroids[:, 1], color='blue', marker='o', s=10, label='Centroids')

    # 设置边界
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    plt.title("CVT with Voronoi Diagram")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()


def test_and_plot_cvt():
    k = 10  # 质心数量
    dim = 2  # 质心维度
    samples = 10000  # 用于生成质心的样本数
    bounds = [[0, 1], [0, 1]]  # 设置 Voronoi 图的边界范围

    # 删除缓存文件以重新生成质心
    fname = centroids_filename(k, dim)
    if Path(fname).is_file():
        print(f"缓存文件 {fname} 已存在，先删除以测试计算过程。")
        Path(fname).unlink()

    # 生成质心并绘制 Voronoi 图
    centroids = cvt(k, dim, samples, cvt_use_cache=False)
    print("生成的质心:")
    print(centroids)
    # 绘制 Voronoi 图
    plot_voronoi(centroids, bounds)

# 运行测试并绘制 Voronoi 图
test_and_plot_cvt()
