import numpy as np
from utils.features import prepare_for_training


class LinearRegresion:
    # 初始化函数,对data进行预处理 data为二位数组
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        # 返回处理后数据，平均值，标准差
        (data_processed, features_mean, features_deviation) = prepare_for_training(data, polynomial_degree=0,
                                                                                   sinusoid_degree=0,
                                                                                   normalize_data=True)
        # 更新为处理后的数据
        self.data = data_processed
        self.labels = labels
        self.data_processed = data_processed
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data
        # shape为行和列,shepe[1]为列数(每个样本有多少xi)，即特征值θ的个数
        num_features = self.data.shape[1]
        # 构建一个θ列向量
        self.theta = np.zeros((num_features, 1))

    #训练函数，alpha学习率(步长),num_iterations迭代次数
    def train(self,alpha,num_iterations=500):
