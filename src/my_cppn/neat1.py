import os
import pickle
from itertools import count

import numpy as np
from tqdm import tqdm

from MY_CPPN.genome import EdgeGene, NodeGene, Genome
from MAPELITE.map import Archive


def sava_gen(archive, gen):
    save_dir = f'./genomes_pkl/gen{gen}'
    os.makedirs(save_dir, exist_ok=True)
    # 构建文件路径
    for behavior, genome in archive.items():
        file_path = os.path.join(save_dir, f'genome{behavior}.pkl')
        # 将当前代的基因组保存为 pkl 文件
        with open(file_path, 'wb') as f:
            pickle.dump(genome, f)


class NEAT:

    def __init__(self, config, edge_class=EdgeGene, node_class=NodeGene):
        # count 先把当前这个数字弹出
        self.global_genome_id = count(0)
        self.config = config
        self.edge_class = edge_class
        self.node_class = node_class
        self.cur_nid = count(config.num_inputs + config.num_outputs)
        self.map_archive = Archive(self.config.archive_rows, self.config.archive_rows, is_cvt=self.config.is_cvt,
                                   cvt_file=self.config.cvt_file)

    def config_new(self):
        genome = Genome(self.config)
        genome.id = next(self.global_genome_id)
        return genome

    def populate(self):
        """
        :return: dict{id:genome}
        """
        pops = {}
        for _ in range(self.config.population_size):
            new_genome = self.config_new()
            self.mutate(new_genome)
            pops[new_genome.id] = new_genome
        return pops

    # TODO 稳态人口和指定后代人口可切换
    def reproduce(self):
        new_pop = {}
        for _ in range(self.config.population_size):
            p1, p2 = np.random.choice(list(self.map_archive.archive.values()), 2, True)
            child = Genome.xover(Genome(self.config), p1, p2)
            # child.ancestor = [p1, p2]
            new_id = next(self.global_genome_id)
            child.id = new_id
            self.mutate(child)
            new_pop[new_id] = child
        return new_pop

    def mutate(self, genome):
        genome.mutate_list = []
        # 添加
        f1 = genome.mutate_split(next(self.cur_nid))
        f2 = genome.mutate_add_conn()

        # 扰动
        genome.mutate_activation()
        genome.mutate_bias()
        genome.mutate_weight()

        # 删除
        f3 = genome.mutate_delete_conn()
        f4 = genome.mutate_delete_node()
        genome.mutate_list.append([f1, f2, f3, f4])

    def run(self, f_fitness, num_generations=5000, report_every=100):
        print('begin')
        for _ in tqdm(range(num_generations)):
            new_pops = self.reproduce()
            f_fitness(list(new_pops.values()),self.config)
            # TODO archive的比较可并行化，cell 之间无关，并且cell 可以采用多genome 方式，提升每一代繁衍的利用率。
            for g_id, genome in new_pops.items():
                self.map_archive.add_to_archive(genome)
            if _ % report_every == 0:
                self.map_archive.display_archive()

        # 直接保存archive
        with open("./map_archive_pong.pkl", "wb") as file:
            pickle.dump(self.map_archive.archive, file)
        print(self.cur_nid)
        return self.map_archive
