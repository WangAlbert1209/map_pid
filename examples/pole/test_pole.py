"""
Test the performance of the best genome produced by evolve-feedforward.py.
"""

import pickle

from CPPN.nn import NNGraph
from cart_pole import CartPole, discrete_actuator_force
from movie import make_movie
from neat1 import NEAT
from utils.config import Config

# load the winner
with open('winner-feedforward', 'rb') as f:
    c = pickle.load(f)

print('Loaded genome:')
print(c)

# Load the config file, which is assumed to live in
# the same directory as this script.
config = Config.load('./config.json')
neat = NEAT(config)

net = NNGraph.from_genome(c)
sim = CartPole()

print()
print("Initial conditions:")
print("        x = {0:.4f}".format(sim.x))
print("    x_dot = {0:.4f}".format(sim.dx))
print("    theta = {0:.4f}".format(sim.theta))
print("theta_dot = {0:.4f}".format(sim.dtheta))
print()

# Run the given simulation for up to 120 seconds.
balance_time = 0.0
while sim.t < 120.0:
    inputs = sim.get_scaled_state()
    action = net.eval(inputs)

    # Apply action to the simulated cart-pole
    force = discrete_actuator_force(action)
    sim.step(force)

    # Stop if the network fails to keep the cart within the position or angle limits.
    # The per-run fitness is the number of time steps the network can balance the pole
    # without exceeding these limits.
    if abs(sim.x) >= sim.position_limit or abs(sim.theta) >= sim.angle_limit_radians:
        break

    balance_time = sim.t

print('Pole balanced for {0:.1f} of 120.0 seconds'.format(balance_time))

print()
print("Final conditions:")
print("        x = {0:.4f}".format(sim.x))
print("    x_dot = {0:.4f}".format(sim.dx))
print("    theta = {0:.4f}".format(sim.theta))
print("theta_dot = {0:.4f}".format(sim.dtheta))
print()
print("Making movie...")
make_movie(net, discrete_actuator_force, 15.0, "feedforward-movie_01.mp4")
