import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

# 基于KMeans对半监督学习的一种解决方案

# 数据处理,切分出训练集和测试集
X_digits, y_digits = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state=42)
# 随机选50个作为标签,准确率仅0.82
n_labels = 50
lgr = LogisticRegression(random_state=42)
lgr.fit(X_train[:n_labels], y_train[:n_labels])
print(lgr.score(X_test, y_test))
# 使用kmeans算法聚类出50个簇,用距离这些中心点最近的样本点作为标签,准确率0.92
k = 50
kms = KMeans(n_clusters=k, random_state=42)
X_digits_dist = kms.fit_transform(X_train)  # 得到1347*50的矩阵,代表每个样本到50个中心点的距离
representative_digits_index = np.argmin(X_digits_dist, axis=0)  # 得到距离每个样本点距离最小的样本编号 50*1
X_train_representative = X_train[representative_digits_index]
y_train_representative = y_train[representative_digits_index]
lgr = LogisticRegression(random_state=42)
lgr.fit(X_train_representative, y_train_representative)
print(lgr.score(X_test, y_test))
# 标签传播进行优化,准确率0.928
y_train_propagated = np.empty(len(X_train), dtype=np.int32)
for i in range(k):
    # 半监督学习,除标签之外其余样本点的y视为未知
    # 将属于第i个簇的样本点的y均设为‘作为标签的代表点’的y
    y_train_propagated[kms.labels_ == i] = y_train_representative[i]
lgr.fit(X_train, y_train_propagated)
print(lgr.score(X_test, y_test))
