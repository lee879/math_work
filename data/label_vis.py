import matplotlib.pyplot as plt
import numpy as np

def label_vision(LABEL):

    unique_labels, label_counts = np.unique(LABEL, return_counts=True)

    # 设置中文字体为SimSun（宋体）
    plt.rcParams['font.sans-serif'] = ['SimSun']

    # 创建柱状图
    plt.bar(unique_labels, label_counts)

    # 设置图表标题和轴标签，使用中文标签
    plt.title("标签分布情况")
    plt.xlabel("标签")
    plt.ylabel("样本数量")

    # 设置 x 轴刻度标签
    plt.xticks(unique_labels, labels=[str(label) for label in unique_labels])

    # 显示图表
    plt.show()
#