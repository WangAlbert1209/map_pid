import os
import pickle
from itertools import count
import time

import numpy as np
from tqdm import tqdm
from cmaes import CMA
from multiprocessing import Pool
import multiprocessing

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
    


    def optimize_single_genome(self, args):
        """
        辅助函数：优化单个基因组
        """
        genome, fitness_func, config, sigma, population_size, n_iterations = args
        
        params = genome.get_parameters()
        
        # 如果参数维度小于2，直接返回原始genome
        if len(params) < 2:
            fitness_func(genome, config)  # 确保genome有一个fitness值
            return genome
        
        # 为当前genome创建特定大小的优化器
        optimizer = CMA(
            mean=params,
            sigma=sigma,
            bounds=np.array([[-30, 30]] * len(params)),
            population_size=population_size,
            )

        best_fitness = float('-inf')
        best_solution = None

        for generation in range(n_iterations):
            solutions = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()
                genome.set_parameters(x)
                # reset before simulation 
                genome.reset_pid()
                fitness_func(genome, config)
                solutions.append((x, genome.fitness))
                
                if genome.fitness > best_fitness:
                    best_fitness = genome.fitness
                    best_solution = x.copy()
            
            optimizer.tell(solutions)

        if best_solution is not None:
            genome.set_parameters(best_solution)
        genome.fitness = best_fitness
        
        return genome

    def optimize_genome_cmaes(self, genomes, fitness_func, config,
                            sigma=1,
                            population_size=30,
                            n_iterations=10,
                            n_processes=multiprocessing.cpu_count()):
        """
        并行使用CMAES优化多个genome的参数
        """
        total_start_time = time.time()
        
        # 准备参数
        args_list = [(genome, fitness_func, config, sigma, population_size, n_iterations) 
                    for genome in genomes]
        
        # 使用进程池并行处理
        with Pool(processes=n_processes) as pool:
            results = pool.map(self.optimize_single_genome, args_list)
        
        # Unpack results
        optimized_genomes = results

        return optimized_genomes

    def run(self, fit_p, fit_s, num_generations=5000, report_every=20):
        print('begin')
        optimize_interval=100
        for _ in tqdm(range(num_generations)):
            new_pops = self.reproduce()
            # 根据进化代数动态调整优化频率
            if _ % optimize_interval == 0 and _ != 0 and False:
                print('cmaes optimize start')
                optimized_genomes = self.optimize_genome_cmaes(
                    genomes=new_pops.values(),
                    fitness_func=fit_s,
                    config=self.config
                )
                new_pops={}
                for genome in optimized_genomes:
                    new_pops[genome.key]=genome
                for genome in new_pops.values():
                    self.map_archive.add_to_archive(genome)
                if _ % 100 == 0:
                    optimize_interval=optimize_interval-10
                    optimize_interval=max(40,optimize_interval)
            else:
                 fit_p(list(new_pops.values()), self.config)
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
