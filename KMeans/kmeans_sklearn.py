import warnings
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score #计算轮廓系数

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

warnings.filterwarnings('ignore')
np.random.seed(42)

# 构造5个中心点
blob_centers = np.array(
    [[0.2, 2.3],
     [-1.5, 2.3],
     [-2.8, 1.8],
     [-2.8, 2.8],
     [-2.8, 1.3]])
# 以5个点为圆心分别发散,形成五个簇
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
# X为每个点的特征,y为每个点所属的簇(类别)
X, y = make_blobs(n_samples=2000, centers=blob_centers,
                  cluster_std=blob_std, random_state=7)
# 绘出这些簇e=(无标签)
plt.scatter(X[:, 0], X[:, 1], c=None, s=1)
plt.xlabel("$x_1$", fontsize=14)
plt.ylabel("$x_2$", fontsize=14, rotation=0)
plt.figure(figsize=(8, 4))
plt.show()

# 训练模型并预测
k = 5
kms = KMeans(n_clusters=k, random_state=42)
y_fit = kms.fit_predict(X)
labels = kms.labels_
cluster_centers = kms.cluster_centers_
X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
y_pred = kms.predict(X_new)
diff_distinct = kms.transform(X_new)  # 得到每个样本点到每个中心点的距离
inertia = kms.inertia_  # 评估标准,每个样本与其所在质心的距离之和,越小越好
print(silhouette_score(X,kms.labels_)) #轮廓系数


# 绘制决策边界
def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)


def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=30, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=50, linewidths=50,
                color=cross_color, zorder=11, alpha=1)


def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                 cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom='off')
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft='off')


plt.figure(figsize=(8, 4))
plot_decision_boundaries(kms, X)
plt.show()
