import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression

# 鸢尾花数据集分类 线性决策边界

# 读入并准备数据
data = pd.read_csv('data/iris.csv')
iris_types = ['SETOSA', 'VERSICOLOR', 'VIRGINICA']  # 种类
x_axis = 'petal_length'  # 两个指标
y_axis = 'petal_width'
# x1 = data[x_axis] x2 = x1[data['class'] == 'SETOSA'] 即取类别为iris_type的所有样本的x_axis列作为横坐标
for iris_type in iris_types:
    plt.scatter(data[x_axis][data['class'] == iris_type], data[y_axis][data['class'] == iris_type], label=iris_type)
plt.show()
num_examples = data.shape[0]
x_train = data[[x_axis, y_axis]].values.reshape((num_examples, 2))
y_train = data['class'].values.reshape((num_examples, 1))
max_iterations = 1000
polynomial_degree = 0
sinusoid_degree = 0
# 训练
lrg = LogisticRegression(x_train, y_train, polynomial_degree, sinusoid_degree)
thetas, cost_histories = lrg.train(max_iterations)
plt.plot(range(len(cost_histories[0])), cost_histories[0], label=lrg.unique_labels[0])
plt.plot(range(len(cost_histories[1])), cost_histories[1], label=lrg.unique_labels[1])
plt.plot(range(len(cost_histories[2])), cost_histories[2], label=lrg.unique_labels[2])
plt.show()  # 三次分类的损失值
# 预测
y_train_predictions = lrg.predict(x_train)
precision = np.sum(y_train_predictions == y_train) / num_examples * 100
print(precision)  # 准确率
# 绘制决策边界,第一个特征作为x轴,第二个特征作为y轴
x_min = np.min(x_train[:, 0])
x_max = np.max(x_train[:, 0])
y_min = np.min(x_train[:, 1])
y_max = np.max(x_train[:, 1])
samples = 150
X = np.linspace(x_min, x_max, samples)
Y = np.linspace(y_min, y_max, samples)
# 绘制分类边界
Z_SETOSA = np.zeros((samples, samples))  # Z_SETOSA[i][j]=1即(i,j)处类别为SETOSA
Z_VERSICOLOR = np.zeros((samples, samples))
Z_VIRGINICA = np.zeros((samples, samples))
for x_index, x in enumerate(X):
    for y_index, y in enumerate(Y):
        data = np.array([[x, y]])
        precision = lrg.predict(data)[0][0]
        if precision == 'SETOSA': #按setosa对所有像素点进行分类,分为是或不是
            Z_SETOSA[x_index][y_index] = 1
        elif precision == 'VERSICOLOR':
            Z_VERSICOLOR[x_index][y_index] = 1
        elif precision == 'VIRGINICA':
            Z_VIRGINICA[x_index][y_index] = 1
# 分类边界图上的点
for iris_type in iris_types:
    plt.scatter(
        x_train[(y_train == iris_type).flatten(), 0],
        x_train[(y_train == iris_type).flatten(), 1],
        label=iris_type
    )
#对分的类绘制三条等高线(决策边界)
plt.contour(X, Y, Z_SETOSA)
plt.contour(X, Y, Z_VERSICOLOR)
plt.contour(X, Y, Z_VIRGINICA)
plt.show()
