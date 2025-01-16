import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from plot_utils import configure_matplotlib

# 在绘图之前调用配置函数
configure_matplotlib()

"""
Test the performance of the best genome produced by evolve-feedforward.py.
"""

import os
import pickle
import sys
# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))  # 获取项目根目录
# sys.path.append("/home/ymh/map_pid") 
sys.path.append("/home/rl_user/Workspace/hank/map_pid") 
from src.my_neat.myfeedfoward import MyFeedForwardNetwork
from src.my_neat.mygenome import CustomGenome
from src.my_neat.myconfig import MyConfig
import neat
from cart_pole import CartPole, continuous_actuator_force, discrete_actuator_force
from movie import make_movie
import cart_pole
from src.my_neat.ctrnn import CTRNN

# load the winner
with open('./best_genome.pkl', 'rb') as f:
    c = pickle.load(f)

print('Loaded genome:')
print(c)

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config_pole.ini')
config = MyConfig(CustomGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

# 为了后续多次运行，先创建网络实例，但需要在循环中不断重新初始化 PID
net = MyFeedForwardNetwork.create(c, config)

# 运行一次仿真并返回最终的theta
def run_simulation_once(genome, net):
    # 每次运行前都要对 PID 进行重置
    genome.reset_pid()
    sim = CartPole()

    # print()
    # print("Initial conditions:")
    # print("        x = {0:.4f}".format(sim.x))
    # print("    x_dot = {0:.4f}".format(sim.dx))
    # print("    theta = {0:.4f}".format(sim.theta))
    # print("theta_dot = {0:.4f}".format(sim.dtheta))
    # print()

    balance_time = 0.0
    while sim.t < 60.0:
        inputs = sim.get_scaled_state()
        action = net.activate(inputs)
        force = continuous_actuator_force(action)
        sim.step(force)

        if abs(sim.x) >= sim.position_limit:
            break
        balance_time = sim.t

    # print('Pole balanced for {0:.1f} of 60.0 seconds'.format(balance_time))
    # print()
    # print("Final conditions:")
    # print("        x = {0:.4f}".format(sim.x))
    # print("    x_dot = {0:.4f}".format(sim.dx))
    # print("    theta = {0:.4f}".format(sim.theta))
    # print("theta_dot = {0:.4f}".format(sim.dtheta))
    # print()
    return sim.theta  # 返回最终的 theta

# 新增：运行一次仿真，返回(初始theta, 最终theta)
def run_simulation_once_with_init_final(genome, net):
    genome.reset_pid()
    # net.reset()
    sim = CartPole()
   
    initial_theta = sim.theta

    # 与上面run_simulation_once一致的过程
    balance_time = 0.0
    steps=0
    error_sum=0
    while sim.t < 60.0:
        inputs = sim.get_scaled_state()
        if hasattr(net, 'activate'):
            action = net.activate(inputs)
        else:
            action = net.advance(inputs, sim.time_step, sim.time_step)
        force = continuous_actuator_force(action)
        sim.step(force)
        steps+=1
        previous_error = (sim.theta % (2 * np.pi)) - 0
        if previous_error > np.pi:
            previous_error = previous_error - (2 * np.pi)
        error_sum += abs(previous_error)
        if abs(sim.x) >= sim.position_limit:
            error_sum = 5
            steps=1
            # break
        balance_time = sim.t
    current_fitness = error_sum / steps if steps > 0 else 5
    print(current_fitness)
    return initial_theta, previous_error

# 单次演示，并制作动画（可根据需求是否启用）
final_theta = run_simulation_once(c, net)



time_const = cart_pole.CartPole.time_step
initial_theta_list = []
final_theta_list = []
num_runs_init_final = 10  # 可自行调整次数
for i in range(num_runs_init_final):
    print("第 {} 次运行(返回初始theta与最终theta)：".format(i + 1))
    # 打印pid 的累积状态
    c.reset_pid()
    for n in c.nodes.values():
        if hasattr(n, 'activation_function'):
            print(n.activation_function.__str__())
    net = MyFeedForwardNetwork.create(c, config)
    print("Making movie...")
    make_movie(net, continuous_actuator_force, 10, f"movies/feedforward-movie_0{i}.mp4")
    init_t, final_t = run_simulation_once_with_init_final(c, net)
    initial_theta_list.append(init_t)
    final_theta_list.append(final_t)


# # 绘制初始theta与最终theta的对应关系散点图
# plt.figure(figsize=(6, 4))
# plt.scatter(initial_theta_list, final_theta_list, color='r', marker='o',s=1 ,label='initial theta & final theta')
# plt.title("Multiple runnings initial theta & final theta")
# plt.xlabel("initial theta")
# plt.ylabel("final theta")
# # plt.ylim(-0.01,0.01)
# plt.legend()
# plt.grid(True)
# plt.savefig("initial_theta_vs_final_theta.png")
# plt.show() 
