import os
import pickle
from itertools import count

import numpy as np
from tqdm import tqdm


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
    def __init__(self, config, archive):
        # count 先把当前这个数字弹出
        self.config = config
        self.genome_indexer = count(1)
        self.map_archive = archive
        self.reproduction = config.reproduction_type(
            config.reproduction_config, None, None
        )

    def populate(self):
        """
        :return: dict{id:genome}
        """
        p = self.reproduction.create_new(
            self.config.genome_type, self.config.genome_config, self.config.pop_size
        )
        return p

    # TODO 稳态人口和指定后代人口可切换
    def reproduce(self):
        new_pop = {}
        for _ in range(self.config.pop_size):
            p1, p2 = np.random.choice(list(self.map_archive.archive.values()), 2, True)
            gid = next(self.reproduction.genome_indexer)
            child = self.config.genome_type(gid)
            child.configure_crossover(p1, p2, self.config.genome_config)
            child.mutate(self.config.genome_config)
            new_pop[gid] = child
        return new_pop

    def run(self, f_fitness, num_generations=5000, report_every=20):
        print('begin')
        for _ in tqdm(range(num_generations)):
            new_pops = self.reproduce()
            f_fitness(list(new_pops.values()), self.config)
            for g_id, genome in new_pops.items():
                self.map_archive.add_to_archive(genome)
            if _ % report_every == 0:
                min_fit, max_fit, best_genome, worst_genome = self.map_archive.display_archive()
                print(max_fit, min_fit)
                with open("./best_genome.pkl", "wb") as file:
                    pickle.dump(best_genome, file)

        # 直接保存archive
        with open("map_archive_pole.pkl", "wb") as file:
            pickle.dump(self.map_archive.archive, file)
        return self.map_archive
