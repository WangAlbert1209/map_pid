import matplotlib.pyplot as plt
import numpy as np

# 假设的数据
features = ['f1', 'f2', 'f3', 'f4', 'f5']
shap_values = [np.random.normal(loc, 0.5, 100) for loc in range(-1, 4)]
feature_values = [np.linspace(0, 1, 100) for _ in range(5)]

# 创建一个散点图，其中每个点的纵坐标稍微偏移，以避免重叠
fig, ax = plt.subplots(figsize=(10, 5))
for i, (feature, shap, feat_val) in enumerate(zip(features, shap_values, feature_values)):
    y = np.random.normal(i, 0.1, size=100)  # 在i附近产生小的随机偏移
    sc = ax.scatter(shap, y, c=feat_val, cmap='viridis', edgecolor='none', s=10, vmin=0, vmax=1)

ax.set_yticks(range(len(features)))
ax.set_yticklabels(features)
ax.set_xlabel('Shapley value')
ax.set_title('Feature Impact and Value Distribution')

# 添加颜色条
cbar = plt.colorbar(sc, ax=ax, orientation='vertical')
cbar.set_label('Feature value')

plt.show()