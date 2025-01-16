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
from src.my_neat.ctrnn import CTRNN

simulation_seconds = 1
time_const = cart_pole.CartPole.time_step
# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = CTRNN.create(genome, config,time_const)
    num_samples = 50
    all_fitnesses = []
    worst_inputs = None  # 新增：存储最差表现时的输入
    worst_fitness = float('-inf')  # 新增：追踪最差适应度
    
    for sample in range(num_samples):
        genome.reset_pid()
        net.reset()
        # for node in genome.nodes.values():
        #     if hasattr(node.activation_function, 'reset'):
        #         print(node.activation_function.__str__())
        sim = cart_pole.CartPole(theta=math.pi)
        error_sum = 0
        steps = 0
        
        while sim.t < simulation_seconds:
            inputs = sim.get_scaled_state()
            action = net.advance(inputs, time_const, time_const)
            force = cart_pole.continuous_actuator_force(action)
            sim.step(force)
            
            previous_error = (sim.theta % (2 * np.pi)) - 0
            if previous_error > np.pi:
                previous_error = previous_error - (2 * np.pi)
            error_sum += abs(previous_error)
            steps += 1
            
            if abs(sim.x) >= sim.position_limit:
                error_sum = 5
                steps=1
                break
        
        current_fitness = error_sum / steps if steps > 0 else 5
        all_fitnesses.append(current_fitness)
        
        # 新增：更新最差适应度和对应的输入
        if current_fitness > worst_fitness:
            worst_fitness = current_fitness
        if np.isnan(inputs[0]) or np.isnan(inputs[2]):
            print(inputs)
            print(sim.x,sim.theta,sim.dx,sim.dtheta,sim.position_limit,sim.angle_limit_radians)
    return -worst_fitness, [(len(genome.nodes)-1)/20,(len(genome.connections)-4)/20] # 返回最差适应度和对应的输入状态


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, './config_pole_rnn.ini')
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
    map_archive = neat2.run(pe.evaluate, eval_genome, num_generations=config.gen)
    minf, maxf, best, worst = map_archive.display_archive()
    # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(best, f)


if __name__ == '__main__':
    run()
