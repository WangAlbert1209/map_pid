"""
A parallel version of XOR using neat.parallel.

Since XOR is a simple experiment, a parallel version probably won't run any
faster than the single-process version, due to the overhead of
inter-process communication.

If your evaluation function is what's taking up most of your processing time
(and you should check by using a profiler while running single-process),
you should see a significant performance improvement by evaluating in parallel.

This example is only intended to show how to do a parallel experiment
in neat-python.  You can of course roll your own parallelism mechanism
or inherit from ParallelEvaluator if you need to do something more complicated.
"""

import os

import neat

from CPPN.parallel import ParallelEvaluator
from MAPELITE.map import Archive
from neat2 import NEAT

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


def eval_genome(genome, config):
    """
    This function will be run in parallel by ParallelEvaluator.  It takes two
    arguments (a single genome and the genome class configuration data) and
    should return one float (that genome's fitness).

    Note that this function needs to be in module scope for multiprocessing.Pool
    (which is what ParallelEvaluator uses) to find it.  Because of this, make
    sure you check for __main__ before executing any code (as we do here in the
    last few lines in the file), otherwise you'll have made a fork bomb
    instead of a neuroevolution demo. :)
    """

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    error = 4.0
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = net.activate(xi)
        error -= (output[0] - xo[0]) ** 2
    return error, [len(genome.nodes) / 30., len(genome.connections) / 30.]


def run(config_file):
    # Load configuration.

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    map_archive = Archive(10, 10, is_cvt=True,
                          cvt_file="../MAPELITE/centroids_1000_dim2.dat")

    neat2 = NEAT(config, map_archive)
    # Create the population, which is the top-level object for a NEAT run.
    init_pops = neat2.populate()
    pe = ParallelEvaluator(5, eval_genome)
    pe.evaluate(list(init_pops.values()), config)
    for id, g in init_pops.items():
        neat2.map_archive.add_to_archive(g)
    map_archive = neat2.run(pe.evaluate, num_generations=300)
    minf, maxf, best, worst = map_archive.display_archive()
    print(f"min {minf}, max {maxf}")
    print(f'best {best.fitness} b:{best.behavior}')
    print(f'worst {worst.fitness} b:{worst.behavior}')
    net = neat.nn.FeedForwardNetwork.create(best, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = net.activate(xi)
        print(output)

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_xor.ini')
    run(config_path)
