import numpy as np
from scipy.optimize import minimize
from utils.features import prepare_for_training
from utils.hypothesis import sigmoid


class LogisticRegression:
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=False):
        (data_processed, features_mean, features_deviation) = prepare_for_training(data, polynomial_degree,
                                                                                   sinusoid_degree,
                                                                                   normalize_data)  # 不进行标准化,便于绘制分类边界
        # 更新为处理后的数据
        self.data = data_processed
        self.labels = labels  # n*1
        self.unique_labels = np.unique(labels)  # 分的类别,枚举值
        self.data_processed = data_processed
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data
        # shape为行和列,shepe[1]为列数(每个样本有多少xi)，即特征值θ的个数
        num_features = self.data.shape[1]  # 分类依据的特征数
        num_unique_labels = np.unique(labels).shape[0]  # 分的类数
        self.theta = np.zeros((num_features, num_unique_labels))  # θ的每列代表对不同类型进行二分类时的系数

    def train(self, max_iterations=1000):
        cost_histories = []
        for label_index, unique_label in enumerate(self.unique_labels):  # 每个类别做一次分类
            current_initial_theta = np.copy(self.theta[:, label_index])  # 取该列(该类别)系数
            current_labels = (self.labels == unique_label).astype(float)  # 如果当前标签的类别为本次分类的类别则为1,否则为0 n*1
            (current_theta, cost_history) = self.gradient_descent(current_labels, current_initial_theta, max_iterations)
            self.theta[:, label_index] = current_theta
            cost_histories.append(cost_history)
        return self.theta, cost_histories

    def gradient_descent(self, labels, current_initial_theta, max_iterations):
        cost_history = []
        num_features = self.data.shape[1]
        current_initial_theta = current_initial_theta.reshape(num_features, 1)  # 传过来的是数组,转为一维列向量
        result = minimize(
            # 要优化的目标
            lambda current_theta: self.cost_function(labels, current_theta.reshape(num_features, 1)),  # 对当前损失值进行最小化
            # 初始化的权重参数
            current_initial_theta,
            # 优化策略
            method='CG',
            # 梯度下降迭代计算公式
            jac=lambda current_theta: self.gradient_step(labels, current_theta.reshape(num_features, 1)),
            # 记录结果
            callback=lambda current_theta: cost_history.append(
                self.cost_function(labels, current_theta.reshape(num_features, 1))),
            # 迭代次数
            options={'maxiter': max_iterations}
        )
        if not result.success:
            raise ArithmeticError('Cannot minimize costfunction' + result.message)
        optimized_theta = result.x  # 优化后的theta
        return optimized_theta, cost_history

    def gradient_step(self, labels, theta):
        num_examples = self.data.shape[0]
        predictions = LogisticRegression.hypothesis(self.data, theta)
        labels_diff = predictions - labels  # n*1
        gradients = (1 / num_examples) * np.dot(self.data.T, labels_diff)  # 计算梯度
        return gradients.flatten()  # 将一维列向量转为数组,minimize中点乘需要数组

    def cost_function(self, labels, theta):
        num_examples = self.data.shape[0]
        predictions = LogisticRegression.hypothesis(self.data, theta)
        y_is_set_cost = np.dot(labels[labels == 1].T, np.log(predictions[labels == 1]))  # 计算属于当前类别预测的的损失值 1*n*n*1
        y_is_not_set_cost = np.dot(1 - labels[labels == 0].T, np.log(1 - predictions[labels == 0]))  # 计算不属于当前类别预测的的损失值
        cost = (-1 / num_examples) * (y_is_set_cost + y_is_not_set_cost)  # 本次分类所有样本的损失值
        return cost

    def predict(self, data):
        num_examples = data.shape[0]
        data_processed = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)[
            0]
        prob = LogisticRegression.hypothesis(data_processed, self.theta)  # 得到每个类别的概率值
        max_prob_index = np.argmax(prob, axis=1)  # 每行在列中找最大值(列所在索引)
        class_prediction = np.empty(max_prob_index.shape, dtype=object)
        for index, label in enumerate(self.unique_labels):  # 此处label为类别
            class_prediction[max_prob_index == index] = label  # 列所在索引与枚举索引一致
        return class_prediction.reshape(num_examples, 1)  # 返回列向量

    @staticmethod
    def hypothesis(data, theta):
        predictions = sigmoid(np.dot(data, theta))  # 使用sigmoid函数将预测值映射为概率
        return predictions  # 转为列向量
