import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

# 对比不同核函数

X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

# 对两个不同维度svm进行对比

poly_kernal_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
])

poly100_kernal_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=100, coef0=1, C=5))
])
poly_kernal_svm_clf.fit(X, y)
poly100_kernal_svm_clf.fit(X, y)


# 绘图
def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')


def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)  # 坐标矩阵——横坐标矩阵X中的每个元素，与纵坐标矩阵Y中对应位置元素，共同构成一个点的完整坐标。
    X = np.c_[x0.ravel(), x1.ravel()]  # x0为所有点的横坐标矩阵,x1为所有点的纵坐标矩阵,矩阵形状为100*100
    y_pred = clf.predict(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)  # 等高线


plt.subplot(121)
plot_predictions(poly_kernal_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.title('degree=3')
plt.subplot(122)
plot_predictions(poly100_kernal_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.title('degree=100')
plt.show()
