
import numpy as np
import neat
import os,sys
import tools.utils as utils
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
# 将项目根目录添加到sys.path
sys.path.append(project_root)
import pickle
from MAPELITE.map import Archive
from metamaterial.tools.shape_tools import getshape
from Genome import CustomGenome
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config_mm.ini')
config = neat.Config(CustomGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

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
def triangulation(shapex, shapey):
    num_point_x = 2 * shapex + 1
    num_point_y = 2 * shapey + 1
    # num_squares_x = int((shapex - 1) / 2)
    # num_squares_y = int((shapey - 1) / 2)
    num_squares = int(shapex * shapey)
    Tri = np.zeros((num_squares * 8, 3))
    Index = np.zeros((num_point_y, num_point_x))
    n = 0
    k = 0
    for i in range(num_point_y):
        for j in range(num_point_x):
            Index[i, j] = n
            n += 1
    for ii in range(shapey):
        for jj in range(shapex):
            # i,j is the index of point which is the left top of the square
            i = ii * 2
            j = jj * 2

            # ====================画三角形
            Tri[k, :] = [Index[i, j], Index[i + 1, j], Index[i, j + 1]]
            Tri[k + 1, :] = [Index[i + 1, j], Index[i + 1, j + 1], Index[i, j + 1]]
            Tri[k + 2, :] = [Index[i, j + 1], Index[i + 1, j + 1], Index[i + 1, j + 2]]
            Tri[k + 3, :] = [Index[i, j + 2], Index[i, j + 1], Index[i + 1, j + 2]]
            Tri[k + 4, :] = [Index[i + 1, j], Index[i + 2, j], Index[i + 2, j + 1]]
            Tri[k + 5, :] = [Index[i + 1, j], Index[i + 2, j + 1], Index[i + 1, j + 1]]
            Tri[k + 6, :] = [
                Index[i + 1, j + 1],
                Index[i + 2, j + 1],
                Index[i + 1, j + 2],
            ]
            Tri[k + 7, :] = [
                Index[i + 2, j + 1],
                Index[i + 2, j + 2],
                Index[i + 1, j + 2],
            ]
            k += 8
    return Tri

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
shapex = shapey = 17
sym4_pcds = sym4_pcd(shapex, shapey)
simple_pcd = point_xy(shapex, shapey)
Tri = triangulation(shapex, shapey)
with open('./map_archive_pole.pkl','rb') as f:
    archive=pickle.load(f)
    # map_archive = Archive(10, 10, is_cvt=True,cell_size=10,
    #                       cvt_file="../MAPELITE/centroids_1000_dim2.dat")
    # map_archive.archive=archive
    for k,gs in archive.items():
        for g in gs:
            if g.fitness[1]<0   :
                getshape(config,g,0.5,simple_pcd,Tri,17,17,None,sympcd=sym4_pcds)
    