from neat.activations import ActivationFunctionSet
import types
class i_activation:

    def __init__(self, dt=0.01, limit=None):
        self.dt = dt
        self.integral = 0.0
        self.limit = limit  # 添加积分限幅
        
    def reset(self):
        self.integral = 0.0
    def __call__(self, x):
        self.integral += x * self.dt
        if self.limit:
            self.integral = max(min(self.integral, self.limit), -self.limit)
        return self.integral 
    def __str__(self):
        return f"Integral: {self.integral}"  # 修正字符串返回
    
    
class d_activation:

    def __init__(self, dt=0.01):
        self.dt = dt
        self.prev_x = None  # 初始化为None以标记第一次调用
        
    def reset(self):
        self.prev_x = None
    def __str__(self):
        return f"Previous value: {self.prev_x}"  # 修正字符串返回
    def __call__(self, x):
        if self.prev_x is None:  # 处理第一次调用
            self.prev_x = x
            return 0.0
        
        dx = x - self.prev_x
        self.prev_x = x
        return dx / self.dt
    
    
class InvalidActivationFunction(TypeError):
    pass
def validate_activation(function):
    # 检查是否为类或可调用对象
    if isinstance(function, type):
        # 如果是类，检查是否有__call__方法
        if not hasattr(function, '__call__'):
            raise InvalidActivationFunction("类必须实现__call__方法")
        
        # 创建实例来检查__call__方法的参数
        instance = function()
        if instance.__call__.__code__.co_argcount != 2:  # self + 1个参数
            raise InvalidActivationFunction("类的__call__方法必须只接受一个参数(除self外)")
    else:
        # 原有的函数验证逻辑
        if not isinstance(function,
                        (types.BuiltinFunctionType,
                         types.FunctionType,
                         types.LambdaType)):
            raise InvalidActivationFunction("需要一个函数对象或带有__call__方法的类")

        if function.__code__.co_argcount != 1:
            raise InvalidActivationFunction("函数必须只接受一个参数")
class CustomActivationFunctionSet(ActivationFunctionSet):
    def __init__(self):
        super().__init__()
        # 添加自定义的激活函数
        self.add('i_node', i_activation)
        # 如果使用类定义的激活函数，确保其是可调用的
        self.add('d_node', d_activation)

    def add(self, name, act):
        validate_activation(act)
        self.functions[name] = act
# 1 class act allow ok
# 2 init when add class node ok
# 3 reset when reset genome ok 
