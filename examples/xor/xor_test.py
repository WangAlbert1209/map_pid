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
import pickle

from CPPN.nn import NNGraph
from CPPN.parallel import ParallelEvaluator
from neat1 import NEAT
from utils.config import Config

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

    net = NNGraph.from_genome(genome)
    error = 0
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = net.eval(xi)
        error += (output[0] - xo[0]) ** 2
    return -error, [len(genome.nodes) / 30., len(genome.connections) / 30.]


def run(config_file):
    # Load configuration.
    # # Load the config file, which is assumed to live in
    config = Config.load(config_file)
    neat = NEAT(config)
    # print(multiprocessing.cpu_count())
    pe = ParallelEvaluator(10, eval_genome)

    init_pops = neat.populate()
    pe.evaluate(list(init_pops.values()), config)
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

    net = NNGraph.from_genome(best)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = net.eval(xi)
        print(output)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'xor_config.json')
    run(config_path)
