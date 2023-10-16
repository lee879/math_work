import numpy as np
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance


def detect_outliers_with_pca(data, n_components=3, threshold=3.0):
    """
    使用PCA和马氏距离检测异常值

    参数：
    data (numpy.ndarray): 数据集，形状为 [n_samples, n_features]。
    n_components (int): PCA要保留的主成分数，默认为3。
    threshold (float): 用于检测异常值的马氏距离阈值，默认为3.0。

    返回：
    outliers (numpy.ndarray): 异常值的索引。
    """
    # 进行PCA降维
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)

    # 计算每个数据点的 Mahalanobis 距离
    cov_estimator = EmpiricalCovariance()
    cov_estimator.fit(data_pca)
    mahalanobis_distances = cov_estimator.mahalanobis(data_pca)

    # 根据阈值检测异常值
    outliers = np.where(mahalanobis_distances > threshold)

    return outliers

