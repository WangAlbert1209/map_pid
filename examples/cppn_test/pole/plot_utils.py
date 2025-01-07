import matplotlib.pyplot as plt
import matplotlib as mpl

def configure_matplotlib():
    # 设置中文字体
    # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS
    # 如果是 Linux 系统可以用下面这行
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  
    # 如果是 Windows 系统可以用下面这行
    # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False 