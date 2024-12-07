# Evolve a control/reward estimation network for the OpenAI Gym
# LunarLander-v2 environment (https://gym.openai.com/envs/LunarLander-v2).
# Sample run here: https://gym.openai.com/evaluations/eval_FbKq5MxAS9GlvB7W6ioJkg


import multiprocessing
import os
import pickle
import random
import time

import gym.wrappers
import matplotlib.pyplot as plt
import neat
import numpy as np

from MAPELITE.map import Archive
from origin_neat.neat2 import NEAT

NUM_CORES = multiprocessing.cpu_count()

env = gym.make('LunarLander-v2')

print("action space: {0!r}".format(env.action_space))
print("observation space: {0!r}".format(env.observation_space))


class LanderGenome(neat.DefaultGenome):
    def __init__(self, key):
        super().__init__(key)
        self.discount = None
        self.behavior = None

    def configure_new(self, config):
        super().configure_new(config)
        self.discount = 0.01 + 0.98 * random.random()

    def configure_crossover(self, genome1, genome2, config):
        super().configure_crossover(genome1, genome2, config)
        self.discount = random.choice((genome1.discount, genome2.discount))

    def mutate(self, config):
        super().mutate(config)
        self.discount += random.gauss(0.0, 0.05)
        self.discount = max(0.01, min(0.99, self.discount))

    def distance(self, other, config):
        dist = super().distance(other, config)
        disc_diff = abs(self.discount - other.discount)
        return dist + disc_diff

    def __str__(self):
        return f"Reward discount: {self.discount}\n{super().__str__()}"


def compute_fitness(genome, net, episodes, min_reward, max_reward):
    m = int(round(np.log(0.01) / np.log(genome.discount)))
    discount_function = [genome.discount ** (m - i) for i in range(m + 1)]

    reward_error = []
    for score, data in episodes:
        # Compute normalized discounted reward.
        dr = np.convolve(data[:, -1], discount_function)[m:]
        dr = 2 * (dr - min_reward) / (max_reward - min_reward) - 1.0
        dr = np.clip(dr, -1.0, 1.0)

        for row, dr in zip(data, dr):
            observation = row[:8]
            action = int(row[8])
            output = net.activate(observation)
            reward_error.append(float((output[action] - dr) ** 2))

    return reward_error, [len(genome.nodes) / 30, len(genome.connections) / 30]


class PooledErrorCompute(object):
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.test_episodes = []
        self.generation = 0

        self.min_reward = -200
        self.max_reward = 200

        self.episode_score = []
        self.episode_length = []

    def simulate(self, nets):
        scores = []
        for genome, net in nets:
            observation_init_vals, observation_init_info = env.reset()
            step = 0
            data = []
            while 1:
                step += 1
                if step < 200 and random.random() < 0.2:
                    action = env.action_space.sample()
                else:
                    output = net.activate(observation_init_vals)
                    action = np.argmax(output)

                # Note: done has been deprecated.
                observation, reward, terminated, done, info = env.step(action)
                data.append(np.hstack((observation, action, reward)))

                if terminated:
                    break

            data = np.array(data)
            score = np.sum(data[:, -1])
            self.episode_score.append(score)
            scores.append(score)
            self.episode_length.append(step)

            self.test_episodes.append((score, data))

        print("Score range [{:.3f}, {:.3f}]".format(min(scores), max(scores)))

    def evaluate_genomes(self, genomes, config):
        self.generation += 1
        t0 = time.time()
        nets = []
        for  g in genomes:
            nets.append((g, neat.nn.FeedForwardNetwork.create(g, config)))

        print("network creation time {0}".format(time.time() - t0))
        t0 = time.time()

        # Periodically generate a new set of episodes for comparison.
        if 1 == self.generation % 10:
            self.test_episodes = self.test_episodes[-300:]
            self.simulate(nets)
            print("simulation run time {0}".format(time.time() - t0))
            t0 = time.time()

        # Assign a composite fitness to each genome; genomes can make progress either
        # by improving their total reward or by making more accurate reward estimates.
        print("Evaluating {0} test episodes".format(len(self.test_episodes)))
        if self.num_workers < 2:
            for genome, net in nets:
                reward_error, behavior = compute_fitness(genome, net, self.test_episodes, self.min_reward,
                                                         self.max_reward)
                genome.behavior = behavior
                genome.fitness = -np.sum(reward_error) / len(self.test_episodes)
        else:
            with multiprocessing.Pool(self.num_workers) as pool:
                jobs = []
                for genome, net in nets:
                    jobs.append(pool.apply_async(compute_fitness,
                                                 (genome, net, self.test_episodes,
                                                  self.min_reward, self.max_reward)))

                for job,  genome in zip(jobs, genomes):
                    reward_error, behavior = job.get(timeout=None)
                    genome.fitness = -np.sum(reward_error) / len(self.test_episodes)
                    genome.behavior = behavior

        print("final fitness compute time {0}\n".format(time.time() - t0))


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_lander.ini')
    config = neat.Config(LanderGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    map_archive = Archive(10, 10, is_cvt=True,
                          cvt_file="../MAPELITE/centroids_1000_dim2.dat")
    neat2 = NEAT(config, map_archive)
    # Create the population, which is the top-level object for a NEAT run.
    init_pops = neat2.populate()
    ec = PooledErrorCompute(5)
    ec.evaluate_genomes(list(init_pops.values()), config)
    for id, g in init_pops.items():
        neat2.map_archive.add_to_archive(g)
    # Run until the winner from a generation is able to solve the environment
    # or the user interrupts the process.

    while 1:
        try:
            neat2.run(ec.evaluate_genomes, num_generations=5)

            # print(gen_best)

            plt.plot(ec.episode_score, 'g-', label='score')
            plt.plot(ec.episode_length, 'b-', label='length')
            plt.grid()
            plt.legend(loc='best')
            plt.savefig("scores.svg")
            plt.close()

            minf, maxf, best, worst = map_archive.display_archive()
            # Use the best genomes seen so far as an ensemble-ish control system.
            best_genomes = best
            best_networks = []
            best_networks.append(neat.nn.FeedForwardNetwork.create(g, config))

            solved = True
            best_scores = []
            for k in range(100):
                observation_init_vals, observation_init_info = env.reset()
                score = 0
                step = 0
                while 1:
                    step += 1
                    # Use the total reward estimates from all five networks to
                    # determine the best action given the current state.
                    votes = np.zeros((4,))
                    for n in best_networks:
                        output = n.activate(observation_init_vals)
                        votes[np.argmax(output)] += 1

                    best_action = np.argmax(votes)
                    # Note: done has been deprecated.
                    observation, reward, terminated, done, info = env.step(best_action)
                    score += reward
                    env.render()
                    if terminated:
                        break

                ec.episode_score.append(score)
                ec.episode_length.append(step)

                best_scores.append(score)
                avg_score = sum(best_scores) / len(best_scores)
                print(k, score, avg_score)
                if avg_score < 200:
                    solved = False
                    break

            if solved:
                print("Solved.")

                # Save the winners.
                for n, g in enumerate(best_genomes):
                    name = 'winner-{0}'.format(n)
                    with open(name + '.pickle', 'wb') as f:
                        pickle.dump(g, f)
                break
        except KeyboardInterrupt:
            print("User break.")
            break

    env.close()


if __name__ == '__main__':
    run()
