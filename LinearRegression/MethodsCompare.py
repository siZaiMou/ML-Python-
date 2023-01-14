import matplotlib as matplotlib  # 绘图
import numpy as np
import os
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from LinearRegression import LinearRegression as LR

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

# 构造数据
X = 2 * np.random.rand(100, 1)  # 构造100*1 X
y = 4 + 3 * X + np.random.rand(100, 1)  # 构造y y=3*X+4+随机抖动
plt.plot(X, y, 'b.')
plt.xlabel('X_1')
plt.ylabel('y')
plt.axis([0, 2, 0, 15])
plt.show()

# 最小二乘法直接求解θ
# 求θ
X_b = np.c_[np.ones((100, 1)), X]  # X多加一列1,用于处理偏置值
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)  # 最小二乘法求最优θ=(X^T.X)^-1.X^T.y
print('leastsquare')
print(theta_best)
# 预测
X_new = np.array([[0], [2]])  # 两个样本进行预测
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)
plt.plot(X_new, y_predict, 'r--')
plt.plot(X, y, 'b.')
plt.axis([0, 2, 0, 15])
plt.show()

# sklearn的线性回归拟合与最小二乘法比较
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print('sklearn.fit')
print(lin_reg.intercept_)  # 偏置参数 与1这一列相乘的θ
print(lin_reg.coef_)  # 权重参数 θ

# 使用批量梯度下降进行预测
lr = LR(data=X, labels=y)
theta = lr.train(0.01, 100)[0]
print('gradient all')
print(theta)

# 随机梯度下降
theta_path_sgd = []
m = len(X_b)
np.random.seed(42)
n_epochs = 50
theta = np.random.randn(2, 1)
t = 0
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index + 1]
        yi = y[random_index:random_index + 1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = 5 / (50 + t)  # 学习率衰减
        theta = theta - eta * gradients
        theta_path_sgd.append(theta)
print('random gradient')
print(theta)

# minibatch小批量梯度下降
theta_path_mgd = []
n_iterations = 50
minibatch = 16  # 小批量个数
theta = np.random.randn(2, 1)
n_epochs = 50
m = len(X_b)
t = 0
for epoch in range(n_epochs):
    shuffled_indecies = np.random.permutation(m)  # 每次随机选取10个样本
    X_b_shuffled_minibatch = X_b[shuffled_indecies]
    y_b_shuffled_minibatch = y[shuffled_indecies]
    for i in range(0, m, minibatch):
        t += 1
        xi = X_b_shuffled_minibatch[i:i + minibatch]
        yi = y_b_shuffled_minibatch[i:i + minibatch]
        gradients = 2 / minibatch * xi.T.dot(xi.dot(theta) - yi)
        eta = 5 / (50 + t)  # 学习率衰减
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)
print('minibatch')
print(theta)
