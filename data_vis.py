'''
author:lee xing
data:2023.10.12
task:2020 Mathematical modeling (Engineering mathematics work)
'''

from data import ptu, vis, wing
import matplotlib.pyplot as plt
import numpy as np

data_ptu, name_ptu = ptu.data_ptu()
data_vis, name_vis = vis.data_vis()
data_wing, name_wing = wing.data_wing()

# 创建一个4x3的子图布局
fig, axs = plt.subplots(4, 3, figsize=(24, 8))

# 定义要显示的子图的索引
show_indices = [
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0)
]

# 遍历所有子图并决定哪些需要显示
for i, j in show_indices:
    p_label = axs[i, j]

    if (i, j) in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]:
        # PTU vision
        index = show_indices.index((i, j))
        p_label.scatter(np.arange(len(data_ptu[:, index])), data_ptu[:, index], s=10, alpha=0.5)
        p_label.set_xlabel('Sampling_Times')
        p_label.set_ylabel("Sensor_data")
        p_label.set_title(name_ptu[index])
    elif (i, j) == (2, 0):
        # VIS vision
        p_label.scatter(np.arange(len(data_vis[:, 0])), data_vis[:, 0], s=10, alpha=0.5)
        p_label.set_xlabel('Sampling_Times')
        p_label.set_ylabel("Sensor_data")
        p_label.set_title(name_vis[0])
    else:
        # Wing vision
        index = show_indices.index((i, j)) - 7
        p_label.scatter(np.arange(len(data_wing[:, index])), data_wing[:, index], s=10, alpha=0.5)
        p_label.set_xlabel('Sampling_Times')
        p_label.set_ylabel("Sensor_data")
        p_label.set_title(name_wing[index])

# 关闭不需要的子图
for i in range(4):
    for j in range(3):
        if (i, j) not in show_indices:
            axs[i, j].axis('off')

plt.tight_layout()
plt.show()








