import numpy as np
from utils.features import prepare_for_training


# 线性回归,批量梯度下降

class LinearRegression:
    # 初始化函数,对data进行预处理 data为二位数组即每个样本的输入值(x) labels为每个样本的y
    # polynomial_degree将特征映射为多个维度，sinusoid_degree进行sin非线性变换，用于非线性回归
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        # 返回处理后数据，平均值，标准差
        # 标准化: (x-μ)/σ 各自维度减各自均值
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
        self.theta = np.random.randn(num_features, 1)

    # 训练函数，alpha学习率(步长),num_iterations迭代次数,执行梯度下降
    def train(self, alpha, num_iterations=500):
        cost_history = self.gradient_descent(alpha=alpha, num_iterations=num_iterations)
        return self.theta, cost_history  # 返回theta和损失值

    # 梯度下降,循环执行每次更新
    def gradient_descent(self, alpha, num_iterations=500):
        cost_history = []  # 损失值
        for i in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data, self.labels))
        return cost_history

    # 参数更新的计算步骤(推导结果)
    def gradient_step(self, alpha):
        num_examples = self.data.shape[0]  # 样本个数
        prediction = LinearRegression.hypothesis(self.data, self.theta)
        # 预测值减真实值得到偏差(y即label标签有监督学习) ,此时delta为列向量,delta为公式中的hθ(x)-y
        delta = prediction - self.labels
        # theta也为列向量
        theta = self.theta
        # np.dot(delta.T,self.data)得到多个样本的预测值(此时为一列),即公式中的delta*xj(xj为data一列)
        self.theta = theta - alpha * (1 / num_examples) * (np.dot(delta.T, self.data)).T  # 乘法计算完再次转置后得到列向量

    # 预测函数(概率密度函数中关于x和θ的函数)静态方法
    @staticmethod
    def hypothesis(data, theta):
        prediction = np.dot(data, theta)  # 矩阵乘法 n*m*m*1=n*1
        return prediction

    # 损失函数，计算损失值
    def cost_function(self, data, labels):
        num_examples = data.shape[0]
        delta = LinearRegression.hypothesis(data, self.theta) - labels
        cost = (1 / 2) * np.dot(delta.T, delta)  # 计算损失值,cost是内积,cost类型为numpy.ndarray
        return cost[0][0]  # 返回每次cost的数值

    # 训练后得到测试(预测)数据的损失值
    def get_cost(self, data, labels):
        data_processed = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)[
            0]
        return self.cost_function(data_processed, labels)

    # 训练后得到测试(预测)数据的预测结果
    def predict(self, data):
        data_processed = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)[
            0]
        predictions = LinearRegression.hypothesis(data_processed, self.theta)
        return predictions
