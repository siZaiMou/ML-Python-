import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mping
import math
import NeuralNetwork.MultilayerPerception as MP

data = pd.read_csv('data/mnist-demo.csv')  # 格式:第一列标签,其它列特征(784个像素点) 每张图片为一个样本,每个样本都有784个像素点
# 展示data
num_display = 25
num_cells = math.ceil(math.sqrt(num_display))
plt.figure(figsize=(10, 10))
for plot_index in range(num_display):
    digit = data[plot_index:plot_index + 1].values
    digit_label = digit[0][0]
    digit_pixels = digit[0][1:]
    image_size = int(math.sqrt(digit_pixels.shape[0]))  # 图片大小(像素点数开根号)
    frame = digit_pixels.reshape((image_size, image_size))
    plt.subplot(num_cells, num_cells, plot_index + 1)
    plt.imshow(frame, cmap='Greys')
    plt.title(digit_label)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()

train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)
train_data = train_data.values
test_data = test_data.values

num_training_examples = 100
X_train = train_data[:num_training_examples, 1:]
y_train = test_data[:num_training_examples, [0]]
X_test = test_data[:, 1:]
y_test = test_data[:, [0]]

layers = [784, 25, 10]
normalize_data = True
max_iterations = 500
alpha = 0.03
ann = MP.MultilayerPerception(X_train, y_train, layers, normalize_data)
(thetas, cost_history) = ann.train(max_iterations, alpha)
y_train_predictions = ann.predict(X_train)
y_test_predictions = ann.predict(X_test)
accuracy_train = np.sum(y_train_predictions == y_train) / y_train.shape[0] * 100
accuracy_test = np.sum(y_test_predictions == y_test) / y_test.shape[0] * 100
print('训练集准确率 ', accuracy_train)
print('测试集准确率 ', accuracy_test)
