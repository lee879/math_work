def replace_outliers_with_mean(data, outliers):
    """
    将异常位置的数据替换为前后索引的均值

    参数：
    data (numpy.ndarray): 数据集，形状为 [n_samples, n_features]。
    outliers (numpy.ndarray): 异常值的索引。

    返回：
    data_fixed (numpy.ndarray): 替换异常值后的数据集。
    """
    data_fixed = data.copy()  # 创建一个副本以保护原始数据

    for outlier_index in outliers[0]:
        # 获取异常值的索引
        if outlier_index > 0:
            prev_index = outlier_index - 1
        else:
            prev_index = outlier_index

        if outlier_index < data.shape[0] - 1:
            next_index = outlier_index + 1
        else:
            next_index = outlier_index

        # 计算前后索引的均值
        mean_value = (data[prev_index] + data[next_index]) / 2.0

        # 替换异常位置的数据为均值
        data_fixed[outlier_index] = mean_value

    return data_fixed