"""
Single-pole balancing experiment using a feed-forward neural network.
"""
import pickle

import neat

import cart_pole
from CPPN.nn import NNGraph
from CPPN.parallel import ParallelEvaluator
from neat1 import NEAT
from utils.config import Config

runs_per_net = 5
simulation_seconds = 60.0


# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = NNGraph.from_genome(genome)

    fitnesses = []

    for runs in range(runs_per_net):
        sim = cart_pole.CartPole()

        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        while sim.t < simulation_seconds:
            inputs = sim.get_scaled_state()
            action = net.eval(inputs)

            # Apply action to the simulated cart-pole
            force = cart_pole.continuous_actuator_force(action)
            sim.step(force)

            # Stop if the network fails to keep the cart within the position or angle limits.
            # The per-run fitness is the number of time steps the network can balance the pole
            # without exceeding these limits.
            if abs(sim.x) >= sim.position_limit or abs(sim.theta) >= sim.angle_limit_radians:
                break

            fitness = sim.t

        fitnesses.append(fitness)
    # print(sim.theta)
    # The genome's fitness is its worst performance across all runs.
    return min(fitnesses), [(sim.x + sim.position_limit) / (2 * sim.position_limit),
                            (sim.theta + sim.angle_limit_radians) / (2 * sim.angle_limit_radians)]


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    # # Load the config file, which is assumed to live in
    config = Config.load('./config.json')
    neat = NEAT(config)
    # print(multiprocessing.cpu_count())
    pe = ParallelEvaluator(10, eval_genome)

    init_pops = neat.populate()
    pe.evaluate(list(init_pops.values()),config)
    for id, g in init_pops.items():
        neat.map_archive.add_to_archive(g)
    # 进化过程
    map_archive = neat.run(pe.evaluate, num_generations=config.gen)
    minf, maxf, best, worst = map_archive.display_archive()
    print(f"min {minf}, max {maxf}")
    print(f'best {best.fitness} b:{best.behavior}')
    print(f'worst {worst.fitness} b:{worst.behavior}')
    best.draw()
    worst.draw()
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(best, f)
if __name__ == '__main__':

    run()
