import os
import pickle
from itertools import count
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset

from MAPELITE.BD_projector import Autoencoder


def sava_gen(archive, gen):
    save_dir = f"./genomes_pkl/gen{gen}"
    os.makedirs(save_dir, exist_ok=True)
    # 构建文件路径
    for behavior, genome in archive.items():
        file_path = os.path.join(save_dir, f"genome{behavior}.pkl")
        # 将当前代的基因组保存为 pkl 文件
        with open(file_path, "wb") as f:
            pickle.dump(genome, f)


def train_vae(vae_model, vae_optimizer, new_pops, device="cpu"):
    """
    训练VAE模型，更新VAE的参数
    :param vae_model: VAE模型
    :param vae_optimizer: 优化器
    :param vae_loss_fn: VAE损失函数
    :param new_pops: 新的种群数据，通常是genome对象，包含vector属性
    :param device: 训练设备 (默认为'cpu'，可设为'cuda'用于GPU训练)
    """
    loss_fn = nn.MSELoss()
    outputs = [torch.tensor(g.output.flatten(), dtype=torch.float32) for g in new_pops]
    data_tensor = torch.stack(outputs).to(device)  # 转为 PyTorch tensor
    dataset = TensorDataset(data_tensor)  # 创建 TensorDataset
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)  # 创建 DataLoader
    vae_model.train()
    for i in range(25):
        for batch in dataloader:
            batch = batch[0].to(device)
            batch = batch.view(-1, 35 * 35)

            vae_optimizer.zero_grad()
            loss = loss_fn(batch, vae_model(batch)[1])
            loss.backward()
            vae_optimizer.step()
            print(i, loss)


def bd(vae_model, new_pops):
    with torch.no_grad():  # 在此阶段我们不需要计算梯度
        vae_model.eval()  # 设置模型为评估模式
        for genome in new_pops:
            # 获取基因组的vector
            input_vector = (
                torch.tensor(genome.output.flatten(), dtype=torch.float32)
                .unsqueeze(0)
                .to("cpu")
            )  # 转为Tensor并加上batch维度
            # 获取VAE模型的重构结果
            latent, recon = vae_model(input_vector)
            # print(latent)
            genome.behavior = latent[0]


class NEAT:
    def __init__(self, config, archive, vae, optimizer):
        # count 先把当前这个数字弹出
        self.config = config
        self.genome_indexer = count(1)
        self.map_archive = archive
        self.reproduction = config.reproduction_type(
            config.reproduction_config, None, None
        )
        self.vectors = self.reference_points(M=2, p=50)
        self.vae = vae
        self.optimizer = optimizer

    def populate(self):
        """
        :return: dict{id:genome}
        """
        p = self.reproduction.create_new(
            self.config.genome_type, self.config.genome_config, self.config.pop_size
        )
        return p

    def reference_points(self, M, p):

        def generator(r_points, M, Q, T, D):
            points = []
            if D == M - 1:
                r_points[D] = Q / (1.0 * T)
                points.append(r_points)
            elif D != M - 1:
                for i in range(Q + 1):
                    r_points[D] = i / T
                    points.extend(generator(r_points.copy(), M, Q - i, T, D + 1))
            return points

        ref_points = np.array(generator(np.zeros(M), M, p, p, 0))
        print(f"generate {len(ref_points)} refs")
        return ref_points

    def adaptation(self, pop, vectors, vectors_, M):
        fits_all = [g.fitness for g in pop]
        # fits_all = [g.fitnesses for g in pop.values()]
        fits_all = np.array(fits_all)
        z_min = np.min(fits_all, axis=0)
        z_max = np.max(fits_all, axis=0)
        vectors = vectors_ * (z_max - z_min)
        vectors = vectors / (np.linalg.norm(vectors, axis=1).reshape(-1, 1))
        neighbours = self.nearest_vectors(vectors)
        return vectors, neighbours

    def nearest_vectors(self, weights):
        sorted_cosine = -np.sort(-np.dot(weights, weights.T), axis=1)
        arccosine_weights = np.arccos(np.clip(sorted_cosine[:, 1], 0, 1))
        return arccosine_weights

    def select_child(self, pop, M, vectors, neighbors, alpha, t, t_max):
        # 找到{min fi}=zmin
        all_f = [g.fitness for g in pop]
        all_f = np.array(all_f)
        z_min = np.min(all_f, axis=0)
        translate_f = all_f - z_min
        cos = np.dot(translate_f, vectors.T) / (
            np.linalg.norm(translate_f, axis=1).reshape(-1, 1) + 1e-21
        )
        arc_c = np.arccos(np.clip(cos, 0, 1))
        idx = np.argmax(cos, axis=1)
        niche = dict(zip(np.arange(vectors.shape[0]), [[]] * vectors.shape[0]))
        idx_u = set(idx)
        for i in idx_u:
            niche.update({i: list(np.where(idx == i)[0])})
        idx_ = []
        all_gs = [g for g in pop]
        all_gs = np.array(all_gs)
        for i in range(0, vectors.shape[0]):
            if len(niche[i]) != 0:
                individual = niche[i]
                arc_c_ind = arc_c[individual, i]
                arc_c_ind = arc_c_ind / neighbors[i]
                d = np.linalg.norm(translate_f[individual, :], axis=1) * (
                    1 + M * ((t / t_max) ** alpha) * arc_c_ind
                )
                idx_adp = np.argmin(d)
                idx_.append(individual[idx_adp])
        select_gs = all_gs[idx_]
        return select_gs

    # TODO 稳态人口和指定后代人口可切换
    def reproduce(self):
        def select_population(archive, population_size):
            # 获取所有的 cell 位置
            cell_positions = list(archive.archive.keys())
            # 选择 population_size 个 cell，通过索引选择
            cell_indices = np.random.choice(
                len(cell_positions), population_size, replace=True
            )

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

        new_pop = {}
        select_populations = select_population(self.map_archive, self.config.pop_size)
        for _ in range(self.config.pop_size):
            p1, p2 = np.random.choice(select_populations, 2, True)
            gid = next(self.reproduction.genome_indexer)
            child = self.config.genome_type(gid)
            child.configure_crossover(p1, p2, self.config.genome_config)
            child.mutate(self.config.genome_config)
            new_pop[gid] = child
        return new_pop

    def initialize_vae(self):
        print("reset autoencoder")
        #  假设 VAE 是您定义的模型类，您可以根据实际情况修改输入和隐层参数
        nput_dim = 784  # 输入尺寸（例如28x28图像）
        atent_dim = 20  # 隐层尺寸
        vae_model = Autoencoder(35 * 35, 2)  # 创建新的VAE实例

        #  如果有GPU，确保模型在正确的设备上
        vae_model = vae_model.to("cpu")
        return vae_model

    def reset_optimizer(self, vae_model, lr=1e-3):
        # 重新初始化优化器
        print("reset optimizer")
        optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-3)
        return optimizer

    def run(self, f_fitness, num_generations=5000, report_every=10):
        print("begin")
        vectors_ = np.copy(self.vectors)
        neighbours = self.nearest_vectors(vectors_)
        # 基于centroid
        centroid = self.map_archive.centroid
        cell_vectors = {tuple(b): vectors_ for b in centroid}
        cell_neighbours = {tuple(b): neighbours for b in centroid}
        # 初始化训练间隔和衰减因子
        initial_training_interval = 20
        decay_factor = 1.1  # 每隔一定的迭代数增大训练间隔
        min_training_interval = 5  # 最小训练间隔
        training_interval = initial_training_interval
        acc_iter = 0
        for _ in range(num_generations):
            acc_iter += 1
            print(f"==============gen {_}==============")
            # reproduction
            new_pops = self.reproduce()
            f_fitness(list(new_pops.values()), self.config)
            bd(self.vae, list(new_pops.values()))
            for genome in new_pops.values():
                self.map_archive.add_to_archive(genome)
            # selection
            for b, gs in self.map_archive.archive.items():
                all_pops = gs
                if len(all_pops) <= 1:
                    continue
                # 自己的vectors
                selected_pop = self.select_child(
                    all_pops,
                    2,
                    cell_vectors[b],
                    cell_neighbours[b],
                    2,
                    _,
                    num_generations,
                )
                self.map_archive.archive[b] = list(selected_pop)
            # adaptation (选择后的才能进行训练，避免冗余的数据进来！)
            if acc_iter >= int(training_interval) and _ != 0:
                all_pops = []
                for b, gs in self.map_archive.archive.items():
                    all_pops.extend(gs)
                self.vae = self.initialize_vae()  # 重新创建VAE模型
                self.optimizer = self.reset_optimizer(self.vae)  # 重新初始化优化器
                train_vae(self.vae, self.optimizer, all_pops)
                # 重新分配位置！！！
                bd(self.vae, all_pops)
                self.map_archive.archive = {}
                for genome in all_pops:
                    self.map_archive.add_to_archive(genome)
                for b, gs in self.map_archive.archive.items():
                    new_vectors, new_neighbours = self.adaptation(
                        list(gs), cell_vectors[b], vectors_, 2
                    )
                    cell_vectors[b] = new_vectors
                    cell_neighbours[b] = new_neighbours
                print("training complete!")
                acc_iter = 0
                training_interval = max(
                    training_interval * decay_factor, min_training_interval
                )

            if _ % report_every == 0:
                self.map_archive.display_moqd()
                with open("map_archive_pole.pkl", "wb") as file:
                    pickle.dump(self.map_archive.archive, file)

        # 直接保存archive
        with open("map_archive_pole.pkl", "wb") as file:
            pickle.dump(self.map_archive.archive, file)
        return self.map_archive
