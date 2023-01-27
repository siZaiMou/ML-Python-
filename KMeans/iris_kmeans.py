import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from KMeans import KMeans

data = pd.read_csv('data/iris.csv')
iris_types = ['SETOSA', 'VERSICOLOR', 'VIRGINICA']
# 使用两个特征,便于绘图展示,对比已知标签和未知标签
x_axis = 'petal_length'
y_axis = 'petal_width'
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for iris_type in iris_types:
    plt.scatter(data[x_axis][data['class'] == iris_type], data[y_axis][data['class'] == iris_type], label=iris_type)
plt.title('label known')
plt.legend()
plt.subplot(1, 2, 2)
plt.scatter(data[x_axis][:], data[y_axis])
plt.title('label unknown')
plt.show()
# kmeans训练
num_examples = data.shape[0]
X_train = data[[x_axis, y_axis]].values.reshape(num_examples, 2)
num_clusters = 3
max_iterations = 50
kms = KMeans(X_train, num_clusters)
centroids, closest_centroids_ids = kms.train(max_iterations)
print(centroids)
# 对比聚类结果
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for iris_type in iris_types:
    plt.scatter(data[x_axis][data['class'] == iris_type], data[y_axis][data['class'] == iris_type], label=iris_type)
plt.title('label known')
plt.legend()
# 绘制聚类后的各簇
plt.subplot(1,2,2)
for centroid_id, centroid in enumerate(centroids):
    current_examples_index = (closest_centroids_ids == centroid_id).flatten()
    plt.scatter(data[x_axis][current_examples_index],data[y_axis][current_examples_index],label = centroid_id)
# 绘制三个中心点
for centroids_id, centroid in enumerate(centroids):
    plt.scatter(centroids[0], centroids[1], c='black', marker='x')
plt.legend()
plt.title('label kmeans')
plt.show()
