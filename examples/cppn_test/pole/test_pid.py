import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import patches
from simple_pid import PID
from cart_pole import CartPole  # 确保导入的是带有 PID 控制的 CartPole
import random
import copy

class PIDOptimizer:
    def __init__(self, population_size=50, generations=15):
        self.population_size = population_size
        self.generations = generations
        self.bounds = [
            (-300, 300),    # Kp bounds
            (-300, 300),    # Ki bounds
            (-300, 300)     # Kd bounds
        ]
        # 每个参数使用16位二进制编码
        self.bits_per_param = 32
        self.total_bits = self.bits_per_param * len(self.bounds)
    
    def create_individual(self):
        # 创建随机二进制串
        return [random.randint(0, 1) for _ in range(self.total_bits)]
    
    def binary_to_decimal(self, binary, param_index):
        # 提取对应参数的二进制位
        start = param_index * self.bits_per_param
        end = start + self.bits_per_param
        param_binary = binary[start:end]
        
        # 转换为十进制
        decimal = 0
        for bit in param_binary:
            decimal = (decimal << 1) | bit
        
        # 映射到参数范围
        low, high = self.bounds[param_index]
        mapped_value = low + (decimal / ((2**self.bits_per_param) - 1)) * (high - low)
        return mapped_value
    
    def decode_individual(self, binary):
        # 将二进制个体解码为PID参数
        return [self.binary_to_decimal(binary, i) for i in range(len(self.bounds))]
    
    def evaluate_fitness(self, individual):

        pid_params = self.decode_individual(individual)        
        kp, ki, kd = pid_params  # 现在应该只有3个值
        pid = PID(kp, ki, kd, setpoint=0)
        cart_pole = CartPole(x=None,theta=None,dx=None,dtheta=None)
        
        
        total_error = 0
        simulation_steps = 3000
        
        for step in range(simulation_steps):
            error = (cart_pole.theta % (2 * np.pi))
            if error > np.pi:
                error -= (2 * np.pi)
            f = pid(error)
            # 限制力的大小在 [-100, 100] 范围内
            f = np.clip(f, -100, 100)
            cart_pole.step(f)
            
            # 基础误差
            total_error += abs(error)
        
        # 综合考虑误差
        fitness = -total_error
        return fitness
    
    def crossover(self, parent1, parent2):
        # 单点交叉
        crossover_point = random.randint(1, self.total_bits - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child
    
    def mutate(self, individual):
        mutated = copy.deepcopy(individual)
        for i in range(self.total_bits):
            if random.random() < 0.2:  # 1%的突变概率
                mutated[i] = 1 - mutated[i]  # 翻转比特
        return mutated
    
    def optimize(self):
        # 初始化种群
        population = [self.create_individual() for _ in range(self.population_size)]
        best_fitness_ever = float('-inf')
        best_individual_ever = None
        
        for generation in range(self.generations):
            # 评估适应度
            fitness_scores = []
            for ind in population:
                fitness = self.evaluate_fitness(ind)
                fitness_scores.append(fitness)
                
                # 更新历史最佳
                if fitness > best_fitness_ever:
                    best_fitness_ever = fitness
                    best_individual_ever = copy.deepcopy(ind)  # 确保深拷贝
            
            # 选择最佳个体
            sorted_pairs = sorted(zip(fitness_scores, population), key=lambda x: x[0], reverse=True)
            sorted_population = [x for _, x in sorted_pairs]
            
            # 创建新种群，确保包含历史最佳个体
            new_population = [copy.deepcopy(best_individual_ever)]  # 第一个位置放历史最佳
            
            # 从当前种群中选择剩余个体
            elite_count = 19  # 保留前19个精英
            new_population.extend([copy.deepcopy(ind) for ind in sorted_population[:elite_count]])
            
            # 生成新种群其余部分
            while len(new_population) < self.population_size:
                parent1 = random.choice(sorted_population[:int(self.population_size * 0.3)])
                parent2 = random.choice(sorted_population[:int(self.population_size * 0.3)])
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population
            current_best_fitness = max(fitness_scores)
            
            # 这里的current_best_fitness 应该总是 <= best_fitness_ever
            assert current_best_fitness <= best_fitness_ever, "精英保留策略出错！"
            
            print(f"Generation {generation + 1}")
            print(f"Current Best fitness: {current_best_fitness}")
            print(f"Historical Best fitness: {best_fitness_ever}")
            if best_individual_ever:
                best_params = self.decode_individual(best_individual_ever)
                print(f"Best parameters: Kp={best_params[0]:.2f}, Ki={best_params[1]:.2f}, Kd={best_params[2]:.2f}")
            print("-" * 50)
        
        return best_individual_ever

# 修改主函数
def test_cart_pole_pid():
    optimizer = PIDOptimizer()
    best_binary = optimizer.optimize()
    # 解码二进制串得到实际的PID参数
    best_params = optimizer.decode_individual(best_binary)
    print(f"最优PID参数: Kp={best_params[0]:.2f}, Ki={best_params[1]:.2f}, Kd={best_params[2]:.2f}")
    
    # 使用优化后的参数创建PID控制器
    pid = PID(best_params[0], best_params[1], best_params[2], setpoint=0)
    
    cart_pole = CartPole(x=0, theta=150 *math. pi / 180, dx=0, dtheta=0,
                 position_limit=30, angle_limit_radians=360 * math.pi / 180)

    # 记录杆的位置
    pole_x_values = []
    pole_y_values = []
    time_steps = []

    # 创建动画的图表
    fig, ax = plt.subplots()
    ax.set_xlim(-100, 100)  # 根据实际需要调整范围
    ax.set_ylim(-2, 2)

    # 创建车和杆
    cart_width = 2
    cart_height = 1
    cart = patches.Rectangle((0, 0), cart_width, cart_height, color='blue')
    ax.add_patch(cart)
    
    pole, = ax.plot([], [], 'o-', lw=2, color='red')

    # 初始化函数
    def init():
        cart.set_xy((-cart_width / 2, -cart_height / 2))
        pole.set_data([], [])
        return cart, pole

    # 更新函数
    def update(frame):
        error = (cart_pole.theta % (2 * np.pi))
        if error > np.pi:
            error -= (2 * np.pi)
        # 计算PID输出
        f = pid(error)
        # 限制力的大小在 [-100, 100] 范围内
        f = np.clip(f, -100, 100)
        cart_pole.step(f)

        # 记录杆的位置
        pole_x = [cart_pole.x, cart_pole.x + cart_pole.lpole * np.sin(cart_pole.theta)]
        pole_y = [0, cart_pole.lpole * np.cos(cart_pole.theta)]
        pole_x_values.append(pole_x)
        pole_y_values.append(pole_y)
        time_steps.append(cart_pole.t)

        # 更新车的位置
        cart.set_xy((cart_pole.x - cart_width / 2, -cart_height / 2))

        # 更新杆的位置
        pole.set_data(pole_x, pole_y)
        
        return cart, pole

    # 创建动画
    ani = FuncAnimation(fig, update, frames=500, init_func=init, blit=True, interval=10, repeat=False)

    # 展示动画
    plt.title('CartPole PID Control Animation')
    plt.show()

    # 在动画之后，我们可以检查记录的杆的路径
    # 画出杆的位置变化
    plt.figure()
    # 方法1：使用颜色映射
    colors = plt.cm.rainbow(np.linspace(0, 1, len(pole_x_values)))
    for i in range(len(pole_x_values)):
        plt.plot(pole_x_values[i], pole_y_values[i], color=colors[i], lw=1.5)
    
    plt.title("Recorded Pole Positions")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.show()

def test_fitness_consistency():
    optimizer = PIDOptimizer()
    
    # 创建一个固定的测试个体
    test_individual = [100, 0, 10]  # 使用固定的PID参数
    
    print("测试fitness评估的一致性：")
    print(f"测试个体 PID 参数: Kp={test_individual[0]}, Ki={test_individual[1]}, Kd={test_individual[2]}")
    print("\n运行10次评估：")
    
    fitness_results = []
    for i in range(10):  # 将次数减少到10以避免长时间运行
        fitness = optimizer.evaluate_fitness(test_individual)
        fitness_results.append(fitness)
        print(f"第 {i+1} 次运行: fitness = {fitness}")
    
    # 检查所有结果是否相同
    is_consistent = all(abs(x - fitness_results[0]) < 1e-10 for x in fitness_results)
    print("\n结果分析：")
    print(f"评估是否完全一致: {is_consistent}")
    if not is_consistent:
        print(f"最大差异: {max(fitness_results) - min(fitness_results)}")
        print(f"平均值: {sum(fitness_results) / len(fitness_results)}")
        print(f"标准差: {np.std(fitness_results)}")

# 运行测试
if __name__ == "__main__":
    # 根据需要选择运行的测试
    # test_fitness_consistency()
    test_cart_pole_pid()  # 注释掉原来的测试 代码逻辑错误已修正
