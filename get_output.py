import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from CPPN.nn import NNGraph
from utils.image_utils import arr_to_img


def generate_image_from_genome(genome, output_path):
    # 从基因组创建神经网络
    nn = NNGraph.from_genome(genome)

    # 定义图像的形状和输入数据
    img_shape = (512, 512)
    x = np.linspace(-1, 1, img_shape[0])
    y = np.linspace(-1, 1, img_shape[1])
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx ** 2 + yy ** 2)

    inputs = np.stack([xx.ravel(), yy.ravel(), r.ravel(), np.ones_like(r.ravel())])
    inputs = inputs.astype(np.float32)
    outputs = nn.eval(inputs)
    outputs = outputs.reshape(img_shape)

    # 转换并保存图像
    img_arr = outputs * 255.
    img_arr = img_arr.T
    img = arr_to_img(img_arr, normalize=False)
    img.save(output_path)

    print(f"Image saved as '{output_path}'")


def process_all_genomes(input_directory, output_directory):
    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)
    plt.figure()
    # 遍历输入目录中的所有 .pkl 文件
    for filename in os.listdir(input_directory):
        if filename.endswith(".pkl"):
            file_path = os.path.join(input_directory, filename)
            with open(file_path, 'rb') as file:
                print(file)
                genome = pickle.load(file)

                plt.scatter(len(genome.nodes), len(genome.enabled_genes))

                # 生成对应的图像并保存
                output_path = os.path.join(output_directory, f"{filename[:-4]}.png")
                # generate_image_from_genome(genome, output_path)
    plt.show()


gen = 2999
# 设置输入和输出目录
input_directory = f'./genomes_pkl/gen{gen}'
output_directory = f'./generated_images/gen{gen}'

# 处理所有基因组并生成图像
process_all_genomes(input_directory, output_directory)
