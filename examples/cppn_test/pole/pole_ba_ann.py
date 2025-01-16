"""
Single-pole balancing experiment using a feed-forward neural network.
"""
import math
import numpy as np
import multiprocessing
import os
import sys



# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))  # 获取项目根目录

sys.path.append(project_root)  # 将项目根目录添加到 Python 路径
sys.path.append("/home/rl_user/Workspace/hank/map_pid") 


import pickle
import neat
import cart_pole
from src.my_cppn.parallel import ParallelEvaluator
from src.mapelite.map import Archive
from src.my_neat.neat2 import NEAT
from src.my_neat.myconfig import MyConfig
from src.my_neat.mygenome import CustomGenome
from src.my_neat.myfeedfoward import MyFeedForwardNetwork

simulation_seconds = 60

# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = MyFeedForwardNetwork.create(genome, config)
    num_samples = 5  # 修改为5次采样
    all_fitnesses = []
    
    # 生成均匀分布的初始角度
    initial_angles = np.linspace(0, 2*np.pi, 5)
    initial_angles=initial_angles[:4]
    initial_angles=initial_angles%(2*np.pi)
    for i in range(len(initial_angles)):
        if initial_angles[i]>np.pi:
            initial_angles[i]=initial_angles[i]-2*np.pi
    for theta_init in initial_angles:
        genome.reset_pid()
        sim = cart_pole.CartPole(theta=theta_init)  # 使用采样的初始角度
        error_sum = 0
        steps = 0
        
        while sim.t < simulation_seconds:
            # 检查是否0-1
            inputs = sim.get_scaled_state()
            action = net.activate(inputs)
            force = cart_pole.continuous_actuator_force(action)
            sim.step(force)
            
            previous_error = (sim.theta % (2 * np.pi))
            if previous_error > np.pi:
                previous_error = previous_error - (2 * np.pi)

            error_sum += abs(previous_error)
            steps += 1
            
            if abs(sim.x) >= sim.position_limit:
                error_sum += 10
                break
        
        current_fitness = error_sum / steps if steps > 0 else 10
        all_fitnesses.append(current_fitness)
        
    worst_fitness = max(all_fitnesses)  # 使用最差的适应度
    genome.fitness = -worst_fitness
    genome.behavior = [(len(genome.nodes)-1)/20, (len(genome.connections)-4)/20]
    return -worst_fitness, [(len(genome.nodes)-1)/20, (len(genome.connections)-4)/20]


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, './config_pole.ini')
    config = MyConfig(CustomGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    map_archive = Archive(10, 10, is_cvt=True,
                          cvt_file=config.cvt_path)

    neat2 = NEAT(config, map_archive)
    # Create the population, which is the top-level object for a NEAT run.
    init_pops = neat2.populate()
    pe = ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    pe.evaluate(list(init_pops.values()), config)
    for id, g in init_pops.items():
        neat2.map_archive.add_to_archive(g)
    map_archive = neat2.run(pe.evaluate,eval_genome, num_generations=config.gen)
    minf, maxf, best, worst = map_archive.display_archive()
    # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(best, f)


if __name__ == '__main__':
    run()
