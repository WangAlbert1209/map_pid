"""
Single-pole balancing experiment using a feed-forward neural network.
"""

import os
import pickle

import neat

import cart_pole
from CPPN.parallel import ParallelEvaluator
from MAPELITE.map import Archive
from origin_neat.neat2 import NEAT

runs_per_net = 5
simulation_seconds = 60.0


# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for runs in range(runs_per_net):
        sim = cart_pole.CartPole()

        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        while sim.t < simulation_seconds:
            inputs = sim.get_scaled_state()
            action = net.activate(inputs)

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
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_pole.ini')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    map_archive = Archive(10, 10, is_cvt=True,
                          cvt_file="../MAPELITE/centroids_1000_dim2.dat")

    neat2 = NEAT(config, map_archive)
    # Create the population, which is the top-level object for a NEAT run.
    init_pops = neat2.populate()
    pe = ParallelEvaluator(5, eval_genome)
    pe.evaluate(list(init_pops.values()), config)
    for id, g in init_pops.items():
        neat2.map_archive.add_to_archive(g)
    map_archive = neat2.run(pe.evaluate, num_generations=200)
    minf, maxf, best, worst = map_archive.display_archive()
    # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(best, f)


if __name__ == '__main__':
    run()
