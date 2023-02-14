import numpy as np
from utils.features import prepare_for_training
from utils.hypothesis import sigmoid, sigmoid_gradient


# from sklearn.neural_network import _multilayer_perceptron

# 多层感知机(用sigmoid的神经网络) MLP/ANN(人工神经网络)
class MultilayerPerception:
    def __init__(self, data, labels, layers, normalize_data=False):
        data_processed = prepare_for_training(data, normalize_data=normalize_data)
        self.data = data_processed[0]
        self.labels = labels
        # 784应为输入样本的特征数
        self.layers = layers  # 784(输入层28*28*1的手写字体图片,784个像素点) 25(隐层提取25个特征) 10(输出10个类别)
        self.normalize_data = normalize_data
        self.thetas = MultilayerPerception.thetas_init(layers)

    def predict(self,data):
        data_processed = prepare_for_training(data, normalize_data=self.normalize_data)[0]
        num_examples = data_processed.shape[0]
        predictions = MultilayerPerception.feedforward_propagation(data_processed,self.thetas,self.layers)
        return np.argmax(predictions,axis=1).reshape((num_examples,1))

    def train(self, max_iterations=1000, alpha=0.1):
        unrolled_thetas = MultilayerPerception.thetas_unroll(self.thetas)  # 将矩阵拉长为一维向量
        (optimized_thetas, cost_history) = MultilayerPerception.gradient_descent(self.data, self.labels,
                                                                                 unrolled_thetas, self.layers,
                                                                                 max_iterations,
                                                                                 alpha)
        self.thetas = MultilayerPerception.thetas_roll(optimized_thetas, self.layers)
        return self.thetas, cost_history

    @staticmethod
    def thetas_init(layers):
        num_layers = len(layers)  # 层数
        thetas = {}  # 字典,每一层层之间都有参数矩阵
        for layer_index in range(num_layers - 1):
            in_count = layers[layer_index]  # 输入个数
            out_count = layers[layer_index + 1]  # 输出个数
            # y=θx,矩阵形状out*in*in*1=out*1,θ多加一列偏置参数(X处理后加一个特征1)
            thetas[layer_index] = np.random.rand(out_count, in_count + 1) * 0.05
        return thetas

    # 将多个矩阵拉长为一个向量,方便更新thetas
    @staticmethod
    def thetas_unroll(thetas):
        num_thetas_layers = len(thetas)
        unrolled_theta = np.array([])  # 待拼接向量
        for theta_layer_index in range(num_thetas_layers):
            unrolled_theta = np.hstack((unrolled_theta, thetas[theta_layer_index].flatten()))  # 向量拼接
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
                                                      MultilayerPerception.thetas_roll(optimized_thetas, layers),layers)
            cost_history.append(cost)
            thetas_gradient = MultilayerPerception.gradient_step(data, labels, optimized_thetas, layers)
            optimized_thetas = optimized_thetas - alpha * thetas_gradient
        return optimized_thetas, cost_history

    # 计算梯度
    @staticmethod
    def gradient_step(data, labels, optimized_thetas, layers):
        thetas = MultilayerPerception.thetas_roll(optimized_thetas,layers)
        thetas_rolled_gradients = MultilayerPerception.feedback_propagation(data, labels, thetas, layers)
        thetas_unrolled_gradients = MultilayerPerception.thetas_unroll(thetas_rolled_gradients)
        return thetas_unrolled_gradients

    # 后向传播
    @staticmethod
    def feedback_propagation(data, labels, thetas, layers):
        num_layers = len(layers)
        num_examples, num_features = data.shape
        num_labels_types = layers[-1]
        deltas = {}  # 计算每一层对输出结果的差异项,初始化deltas(deltas用来更新theta,和theta同形)
        for layer_index in range(num_layers - 1):
            in_count = layers[layer_index]
            out_count = layers[layer_index + 1]
            deltas[layer_index] = np.zeros((out_count, in_count + 1))
        # 前向传播,计算每个样本每层的输入和输出结果
        for example_index in range(num_examples):
            layers_inputs = {}
            layers_activations = {}
            layer_activation = data[example_index, :].reshape((num_features, 1))
            layers_activations[0] = layer_activation  # 输入的一个样本转为列向量 785*1
            # 逐层进行前向传播
            for layer_index in range(num_layers - 1):
                layer_theta = thetas[layer_index]  # 25*785 10*26
                layer_input = np.dot(layer_theta, layer_activation)  # X为列向量,这里为θ*X而非前向传播函数中的X*θ.T 得到下一层输入(这一层输出) 25*1
                layer_activation = np.vstack((np.array([1]), sigmoid(layer_input)))  # 加一行1(对应前向传播中加一列1)并激活作为下一层的输入激活数据
                layers_inputs[layer_index + 1] = layer_input  # 记录下一层的输入
                layers_activations[layer_index + 1] = layer_activation  # 记录下一层输入激活后结果
            output_layer_activation = layer_activation[1:, :]  # 最终输出结果并去掉偏置参数(对应前向传播中去掉第一列1,这里是去掉第一行1) 10*1
            delta = {}  # delta为误差项,用来计算deltas
            # 标签处理为one-hot形式 处理为10*1
            bitwise_labels = np.zeros((num_labels_types, 1))
            bitwise_labels[labels[example_index][0]] = 1
            # 计算最后一层的delta = output - y,对于一个样本得到10*1的矩阵
            delta[num_layers - 1] = output_layer_activation - bitwise_labels
            # 从倒数第二层往前逐层计算delta
            for layer_index in range(num_layers - 2, 0, -1):
                layer_theta = thetas[layer_index]  # 10*26
                next_delta = delta[layer_index + 1]  # 后一层delta 10*1(已计算出)
                layer_input = layers_inputs[layer_index]  # 前向传播中对每个后一层的输入数据 25*1
                layer_input = np.vstack((np.array([1]), layer_input))  # 加一行1
                # 按公式计算 26*10*10*1*float=26*1
                delta[layer_index] = np.dot(layer_theta.T, next_delta) * sigmoid_gradient(layer_input)
                delta[layer_index] = delta[layer_index][1:, :]  # 去除一行1 得到倒数第二层的delta 25*1
            # 计算梯度值
            for layer_index in range(num_layers - 1):
                # 10*1*1*26 25*1*1*785 用来更新thetas
                layer_delta = np.dot(delta[layer_index + 1], layers_activations[layer_index].T)
                # 每个样本都能计算出delta并用来更新deltas
                deltas[layer_index] = deltas[layer_index] + layer_delta
        # 最终对deltas求平均
        for layer_index in range(num_layers - 1):
            deltas[layer_index] = deltas[layer_index] * 1 / num_examples
        return deltas

    # 计算损失值
    @staticmethod
    def cost_function(data, labels, thetas, layers):
        num_layers = len(layers)
        num_examples = data.shape[0]
        num_labels = layers[-1]  # 输出层种类个数
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
            # 拼接一列1以计算偏置项(每层计算结果作为下一层输入,像data初始化时一样加一列1作为计算偏置项的特征)
            # thetas的偏置参数在init函数中已经处理过
            out_layer_activation = np.hstack((np.ones((num_examples, 1)), out_layer_activation))
            in_layer_activation = out_layer_activation
        # 返回输出层(被赋值给in_layer_activation)结果,结果中不要偏置项的一列1
        return in_layer_activation[:, 1:]
