# 初始化训练间隔和衰减因子
initial_training_interval = 20
decay_factor = 1.1  # 每隔一定的迭代数增大训练间隔
min_training_interval = 5  # 最小训练间隔
sum_i = 0
# 在循环中逐步增加训练间隔
for iteration in range(100):
    training_interval = max(initial_training_interval * (decay_factor ** iteration), min_training_interval)
    sum_i += training_interval
    print(training_interval)
    if training_interval > 200:
        print(iteration, sum_i)
        break
