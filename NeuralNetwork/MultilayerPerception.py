import numpy as np
from utils.features import prepare_for_training
from utils.hypothesis import sigmoid, sigmoid_gradient


# from sklearn.neural_network import _multilayer_perceptron

# 多层感知机
class MultilayerPerception:
    def __init__(self, data, labels, layers, normalize_data=False):
        data_processed = prepare_for_training(data, normalize_data=normalize_data)
        self.data = data_processed
        self.labels = labels
        self.layers = layers  # 784(输入层28*28*1的手写字体图片,784个像素点) 25(隐层提取25个特征) 10(输出10个类别)
        self.normalize_data = normalize_data
        self.thetas = MultilayerPerception.thetas_init(layers)

    def train(self, max_iterations=1000, alpha=0.1):
        unrolled_thetas = MultilayerPerception.thetas_unroll(self.thetas)  # 将矩阵拉长为一维向量
        MultilayerPerception.gradient_descent(self.data, self.labels, unrolled_thetas, self.layers, max_iterations,
                                              alpha)

    @staticmethod
    def thetas_init(layers):
        num_layers = len(layers)  # 层数
        thetas = {}  # 字典,每一层层之间都有参数矩阵
        for layer_index in range(num_layers - 1):
            in_count = layers[layer_index]  # 输入个数
            out_count = layer_index[layer_index + 1]  # 输出个数
            # y=θx,矩阵形状out*in*in*1=out*1,θ多加一列偏置参数(X处理后加一个特征1)
            thetas[layer_index] = np.random.rand(out_count, in_count + 1) * 0.05
        return thetas

    # 将多个矩阵拉长为一个向量
    @staticmethod
    def thetas_unroll(thetas):
        num_thetas_layers = len(thetas)
        unrolled_theta = np.array([])  # 待拼接向量
        for theta_layer_index in range(num_thetas_layers):
            np.hstack(unrolled_theta, thetas[theta_layer_index].flatten())  # 向量拼接
        return unrolled_theta

    # 将向量还原为矩阵
    @staticmethod
    def thetas_roll(unrolled_thetas, layers):
        num_layers = len(layers)
        thetas = {}
        unrolled_shift = 0  # 在向量中变换到的位置
        for layer_index in range(num_layers - 1):
            in_count = layers[layer_index]
            out_count = layers[layer_index + 1]
            thetas_height = out_count
            thetas_width = in_count + 1
            thetas_volume = thetas_height * thetas_width  # 本矩阵大小
            start_shift = unrolled_shift  # 在向量中开始转换位置
            end_shift = start_shift + thetas_volume
            layer_thetas_unrolled = unrolled_thetas[start_shift:end_shift]  # 在向量中截取出转换成本矩阵的部分
            thetas[layer_index] = layer_thetas_unrolled.reshape((thetas_height, thetas_width))
            unrolled_shift += thetas_volume  # 更新变换到的位置
        return thetas

    # 梯度下降
    @staticmethod
    def gradient_descent(data, labels, unrolled_thetas, layers, max_iterations, alpha):
        optimized_thetas = unrolled_thetas
        cost_history = []
        for _ in range(max_iterations):
            cost = MultilayerPerception.cost_function(data, labels,
                                                      MultilayerPerception.thetas_roll(unrolled_thetas, layers))
        theta_gradient = MultilayerPerception.gradient_step(data,labels,optimized_thetas,layers)

    # 梯度下降步骤
    @staticmethod
    def gradient_step(data, labels, optimized_thetas, layers):
        thetas = MultilayerPerception.thetas_roll(optimized_thetas)
        MultilayerPerception.feedback_propagation(data,labels,thetas,layers)

    #计算损失值
    @staticmethod
    def cost_function(data, labels, thetas, layers):
        num_layers = len(layers)
        num_examples = data.shape[0]
        num_labels = layers[-1]  # 输出层个数
        # 前向传播一次,计算预测结果
        predictions = MultilayerPerception.feedforward_propagation(data, thetas, layers)
        bitwise_labels = np.zeros((num_examples, num_labels))
        for example_index in range(num_examples):
            # 传进来的labels只有每个样本的类别,将其转换为样本数*类别数的矩阵,属于哪个类别就赋值1,其它都为0
            # 每个样本的标签都制作为one-hot编码
            bitwise_labels[example_index][labels[example_index][0]] = 1
        # 交叉熵计算损失-p*logp,属于1和不属于1的损失分别计算
        bit_set_cost = np.sum(np.log(predictions[bitwise_labels == 1]))
        bit_not_set_cost = np.sum(np.log(1 - predictions[bitwise_labels == 0]))
        cost = (-1 / num_examples) * (bit_set_cost + bit_not_set_cost)
        return cost

    # 前向传播
    @staticmethod
    def feedforward_propagation(data, thetas, layers):
        num_layers = len(layers)
        num_examples = data.shape[0]
        in_layer_activation = data
        for layer_index in range(num_layers - 1):
            theta = thetas[layer_index]
            # 公式中是θ*X,代码中为X*θ,类似逻辑回归的代码
            # θ*X θ转置以矩阵计算 样本数*特征数*特征数*类别数=样本数*类别数 每个样本对每个类别的得分
            out_layer_activation = sigmoid(np.dot(in_layer_activation, theta.T))
            # 拼接一列1以计算偏置项(每层计算结果作为下一层输入,像data初始化时一样加一列1)
            # thetas的偏置参数在init函数中已经处理过
            out_layer_activation = np.hstack(np.ones((num_examples, 1)), out_layer_activation)
            in_layer_activation = out_layer_activation
        # 返回输出层(被赋值给in_layer_activation)结果,结果中不要偏置项的一列1
        return in_layer_activation[:, 1:]
