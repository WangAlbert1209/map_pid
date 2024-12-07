import numpy as np


# 定义激活函数
def gaussian(x):
    x = np.clip(x, -3.4, 3.4)
    return np.exp(-5.0 * x ** 2)


def exp(x):
    x = np.clip(x, -60.0, 60.0)
    return np.exp(x)


def log(x):
    return np.log(np.maximum(1e-7, x))


def relu(x):
    return np.clip(x, 0., None)


def abs(x):
    return np.abs(x)


def sigmoid(x):
    x = np.clip(x * 5, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    x = np.clip(x * 2.5, -60.0, 60.0)
    return np.tanh(x)


def sin(x):
    x = np.clip(x * 5, -60.0, 60.0)
    return np.sin(x)


def square(x):
    return x ** 2
