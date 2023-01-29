from matplotlib.image import imread
from matplotlib.image import imsave
from sklearn.cluster import KMeans

image = imread('data/ladybug.png')  # 533*800*3的彩色数据
X = image.reshape(-1, 3)  # 将前二维转为一维,作为533*800个样本,特征数为3
kms = KMeans(n_clusters=20, random_state=42).fit(X)
# cluster_centers_为训练得到的所有中心点,labels_为每个样本所属的中心点编号,此操作得到每个样本所属的中心点
# 并将每个样本点代替为所属簇的中心点,还原为图像
segmented_img = kms.cluster_centers_[kms.labels_].reshape(533, 800, 3)
imsave('data/segmented_ladybug.png', segmented_img)
