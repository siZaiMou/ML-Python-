import numpy as np


class KMeans:
    def __init__(self, data, num_clusters):
        self.data = data
        self.num_clusters = num_clusters

    def train(self, max_iterations):
        centroids = self.centroids_init()  # 随机选择k个中心点
        num_examples = self.data.shape[0]
        closest_centroids_ids = np.empty((num_examples, 1))
        for _ in range(max_iterations):
            closest_centroids_ids = self.centroids_find_closest(centroids)  # 得到当前每个样本点到k个中心点的距离,找到最近的中心点(每个样本属于的簇)
            centroids = self.centroids_compute(closest_centroids_ids)  # 更新中心点
        return centroids,closest_centroids_ids

    def centroids_init(self):
        num_examples = self.data.shape[0]
        random_ids = np.random.permutation(num_examples)
        centroids = self.data[random_ids[:self.num_clusters], :]  # 随机取打乱id后的前num_clusters个样本作为中心点
        return centroids

    def centroids_find_closest(self, centroids):
        num_examples = self.data.shape[0]
        num_centroids = centroids.shape[0]
        closest_centroids_ids = np.zeros((num_examples, 1))
        # 计算所有数据到中心点的距离
        for example_index in range(num_examples):
            distance = np.zeros((num_centroids, 1))  # 此样本点到本次所有中心点的距离
            for centroids_index in range(num_centroids):
                distance_diff = self.data[example_index, :] - centroids[centroids_index, :]
                distance[centroids_index] = np.sum(distance_diff ** 2)
            closest_centroids_ids[example_index] = np.argmin(distance)  # 得到此样本距离最近中心点的id
        return closest_centroids_ids

    def centroids_compute(self, closest_centroids_ids):
        num_features = self.data.shape[1]
        centroids = np.zeros((self.num_clusters, num_features))
        for centroids_id in range(self.num_clusters):
            closest_ids = closest_centroids_ids == centroids_id  # 得到属于当前中心点的样本id
            centroids[centroids_id] = np.mean(self.data[closest_ids.flatten(), :], axis=0)  # 计算这些样本点的均值作为新中心点
        return centroids
