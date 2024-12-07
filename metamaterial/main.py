import multiprocessing
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
# 将项目根目录添加到sys.path
sys.path.append(project_root)
import math
import neat
import numpy as np
import tools.constraints as hc
from CPPN.parallel import ParallelEvaluator
from MAPELITE.map import Archive
from fem import getfit
from metamaterial.tools import utils
from metamaterial.tools.mesh_tools import getmesh
from metamaterial.tools.shape_tools import find_contour, triangulation, get_outside_Tri
from metamaterial.neat2 import NEAT,train_vae,bd
from Genome import CustomGenome
from MAPELITE.BD_projector import Autoencoder
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
# pcd
def sym4_pcd(shapex, shapey):
    l = 2 * shapex + 1
    w = 2 * shapey + 1
    x = np.linspace(0, 1, l)
    y = np.linspace(0, 1, w)
    X = np.zeros((w, l))
    Y = np.zeros((w, l))
    X[:, :] = x
    Y[:, :] = y.reshape(-1, 1)
    input_xy = np.stack((X.flatten(), Y.flatten()), axis=1)
    for i in range(2):
        input_xy[:, i] = utils.normalize(input_xy[:, i], 0, 1)  # [-1,1]
    #   第一象限y=x对称
    for i, p in enumerate(input_xy):
        if p[1] >= p[0] and all(p >= 0):
            temp = input_xy[i][0]
            input_xy[i][0] = input_xy[i][1]
            input_xy[i][1] = temp

        # 旋转变换s
        mid_x = (w - 1) / 2
        mid_y = (l - 1) / 2
        last_x = w - 1
        last_y = l - 1
        # 左右对称
        for j in range(l):
            for i in range(w):
                if i < mid_x:
                    input_xy[j * w + i] = input_xy[j * w + last_x - i]
        # 上下对称
        for j in range(w):
            for i in range(l):
                if i < mid_y:
                    input_xy[j + i * w] = input_xy[j + w * (last_y - i)]
    return input_xy
def generate_samples(vae, num_samples=10, device='cpu'):
    # 在评估模式下运行模型
    vae.eval()

    with torch.no_grad():
        # 从标准正态分布采样潜在变量 z
        z = torch.randn(num_samples, 2).to(device)

        # 使用解码器生成样本
        generated = vae.decoder(z)

        # 取解码器输出的均值部分（用于生成图像）
        samples = generated.cpu().numpy()  # samples是一个shape为(num_samples, input_dim)的numpy数组

        return samples



def point_xy(shapex, shapey):
    l = 2 * shapex + 1
    w = 2 * shapey + 1
    x = np.linspace(0, 1, l)
    y = np.linspace(0, 1, w)
    X = np.zeros((w, l))
    Y = np.zeros((w, l))

    X[:, :] = x
    Y[:, :] = y.reshape(-1, 1)
    input_xy = np.stack((X.flatten(), Y.flatten()), axis=1)
    # normalize the input_xyz
    for i in range(2):
        input_xy[:, i] = utils.normalize(input_xy[:, i], 0, 1)  # [-1,1]
    return input_xy


shapex = shapey = 17
sym4_pcd = sym4_pcd(shapex, shapey)
simple_pcd = point_xy(shapex, shapey)
Tri = triangulation(shapex, shapey)


# eval_genomes
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    outputs = []

    for point in sym4_pcd:
        output = net.activate(point)
        outputs.append(output)
    outputs = np.array(outputs)
    outputs = utils.scale(outputs)
    outputs_square = outputs.reshape(2 *shapex + 1, -1)
    Index, X, Y, Cat = find_contour(
        a=outputs_square,
        thresh=0.5,
        pcd=simple_pcd,
        shapex=shapex,
        shapey=shapey,
        pcdtype='sym4',
    )
    x_values = X.flatten()
    y_values = Y.flatten()
    cat_values = Cat.flatten()
    index_values = Index.flatten()
    index_x_y_cat = np.concatenate(
        (
            index_values.reshape(-1, 1),
            x_values.reshape(-1, 1),
            y_values.reshape(-1, 1),
            cat_values.reshape(-1, 1),
        ),
        axis=1,
    )
    # 1,2  1是outside ，2是inside
    outtri = get_outside_Tri(Tri, index_x_y_cat)
    mesh = getmesh(index_x_y_cat, outtri)
    f1, f2, area,solved = getfit(mesh, 'sym4', "Max:E-Min:nu")
    handle_contraints = hc.handle_constarints()
    filtered_tri = mesh.cells()
    handle_contraints.cal_violate_num(Cat,filtered_tri )
    violate_num = handle_contraints.violate_num
    # constraints handling
    
    if not solved or violate_num>0:
        f1 = -99999
        f2=99999
    # return [f1, -f2], [area,(len(genome.nodes)-1)/10]
    return [-f1, f2], [None,None],outputs_square


# experiments
def run_experiment():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_mm.ini')
    config = neat.Config(CustomGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    map_archive = Archive(10, 10, is_cvt=True,cell_size=5,
                          cvt_file="../MAPELITE/centroids_400_dim2.dat")
    device = torch.device('cpu')
    vae=Autoencoder(35*35,2).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    neat2 = NEAT(config, map_archive,vae,optimizer)
    # Create the population, which is the top-level object for a NEAT run.


    init_pops = neat2.populate()

    pe = ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    pe.evaluate(list(init_pops.values()), config)
    train_vae(vae,optimizer,list(init_pops.values()))
    bd(vae,list(init_pops.values()))
    
    # 测试并生成样本
    num_samples = 10
    generated_samples = generate_samples(vae, num_samples=num_samples, device=torch.device('cpu'))

    # 将生成的样本重构为 28x28 图像并展示
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 2))
    for i, ax in enumerate(axes):
        ax.imshow(generated_samples[i].reshape(35, 35), cmap='gray')
        ax.axis('off')
    plt.show()
    plt.savefig('./recon.png')
    plt.close()
    for id, g in init_pops.items():
        neat2.map_archive.add_to_archive(g)
    map_archive = neat2.run(pe.evaluate, num_generations=5000)


run_experiment()
