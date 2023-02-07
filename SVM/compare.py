import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn import datasets

iris = datasets.load_iris()
X = iris['data'][:, (2, 3)]  # 两个特征方便展示
y = iris['target']
setosa_or_versicolor = (y == 0) | (y == 1)  # 只取0或1两个类别
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

svm_clf = SVC(kernel='linear', C=100.0)
svm_clf.fit(X, y)

# 一般模型
x0 = np.linspace(0, 5.5, 200)
pred_1 = 5 * x0 - 20
pred_2 = x0 - 1.8
pred_3 = 0.1 * x0 + 0.5
plt.figure(figsize=(14, 4))
plt.subplot(121)
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'ys')
plt.plot(x0, pred_1, 'g--')
plt.plot(x0, pred_2, 'm--')
plt.plot(x0, pred_3, 'r--')
plt.axis([0, 5.5, 0, 2])


# SVM分类结果
def plot_svm_decision_boundary(svm_clf, xmin, xmax, sv=True):
    w = svm_clf.coef_[0]  # w
    b = svm_clf.intercept_[0]  # b
    x0 = np.linspace(xmin, xmax, 200)
    x1 = -x0 * w[0] / w[1] - b / w[1]  # 决策平面为x0*w0+x1*w1+b=0,构造出x0后可直接得出x1,即决策边界
    margin = 1 / w[1]  # 支持向量距平面距离(此处x1为y轴)
    gutter_up = x1 + margin
    gutter_down = x1 - margin
    if sv:
        svs = svm_clf.support_vectors_  # 支持向量
        plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, x1, 'k-', linewidth=2)  # 决策边界
    plt.plot(x0, gutter_up, 'k-', linewidth=2)  # 决策上界
    plt.plot(x0, gutter_down, 'k-', linewidth=2)  # 决策下界


plt.subplot(122)
plot_svm_decision_boundary(svm_clf, 0, 5.5, True)
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'ys')
plt.axis([0, 5.5, 0, 2])
plt.show()
