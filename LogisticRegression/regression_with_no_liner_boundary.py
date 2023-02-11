import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from LogisticRegression import LogisticRegression

# 非线性决策边界
data = pd.read_csv('data/microchips-tests.csv')
# 类别标签
validities = [0, 1]
# 选择两个特征
x_axis = 'param_1'
y_axis = 'param_2'
# 原始数据散点图
for validity in validities:
    plt.scatter(
        data[x_axis][data['validity'] == validity],
        data[y_axis][data['validity'] == validity],
        label=validity
    )
plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.title('Microchips Tests')
plt.legend()
plt.show()
# 数据处理
num_examples = data.shape[0]
x_train = data[[x_axis, y_axis]].values.reshape((num_examples, 2))
y_train = data['validity'].values.reshape((num_examples, 1))
# 设置训练参数
max_iterations = 100000
regularization_param = 0
polynomial_degree = 5  # 非线性回归的高维次幂,非线性变换后再传回sigmoid函数
sinusoid_degree = 0
# 训练
lgr = LogisticRegression(x_train, y_train, polynomial_degree, sinusoid_degree)
(thetas, costs) = lgr.train(max_iterations)
# 绘制损失曲线
labels = lgr.unique_labels
plt.plot(range(len(costs[0])), costs[0], label=labels[0])
plt.plot(range(len(costs[1])), costs[1], label=labels[1])
plt.xlabel('Gradient Step')
plt.ylabel('Cost')
plt.legend()
plt.show()
# 预测
y_train_predictions = lgr.predict(x_train)
precision = np.sum(y_train_predictions == y_train) / y_train.shape[0] * 100
print(precision)

# 绘制决策边界(非线性)
num_examples = x_train.shape[0]
samples = 150
x_min = np.min(x_train[:, 0])
x_max = np.max(x_train[:, 0])
y_min = np.min(x_train[:, 1])
y_max = np.max(x_train[:, 1])
X = np.linspace(x_min, x_max, samples)
Y = np.linspace(y_min, y_max, samples)
Z = np.zeros((samples, samples))
# 结果展示
for x_index, x in enumerate(X):
    for y_index, y in enumerate(Y):
        data = np.array([[x, y]])
        Z[x_index][y_index] = lgr.predict(data)[0][0]

positives = (y_train == 1).flatten()
negatives = (y_train == 0).flatten()
plt.scatter(x_train[negatives, 0], x_train[negatives, 1], label='0')
plt.scatter(x_train[positives, 0], x_train[positives, 1], label='1')
plt.contour(X, Y, Z)  # 分类结果0,1之间的等高线
plt.xlabel('param_1')
plt.ylabel('param_2')
plt.title('Microchips Tests')
plt.legend()
plt.show()
