import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

# 单特征值回归预测

data = pd.read_csv('data/world-happiness-report-2017.csv')
train_data = data.sample(frac=0.8)  # 训练数据取80%的样本
test_data = data.drop(train_data.index)  # 原始样本删掉训练样本作为测试数据
input_param_name = 'Economy..GDP.per.Capita.'  # 根据"Economy..GDP.per.Capita."列预测"Happiness.Score"
output_param_name = 'Happiness.Score'
x_train = train_data[[input_param_name]].values  # 输入一列数据
y_train = train_data[[output_param_name]].values  # labels
x_test = test_data[[input_param_name]].values
y_test = test_data[[output_param_name]].values

# 绘制预处理后数据(散点图)
plt.scatter(x_train,y_train,label='Train data')
plt.scatter(x_test,y_test,label='Test data')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Processed Data')
plt.legend()
plt.show()

# 训练
num_iterations = 500
learning_rate = 0.01
lr = LinearRegression(x_train, y_train)
(theta, cost_history) = lr.train(learning_rate, num_iterations)

#绘制每次迭代中的cost
plt.plot(range(0,num_iterations), cost_history)
plt.xlabel('num_iterations')
plt.ylabel('cost')
plt.title('Cost per train')
plt.show()

# 绘制回归线
prediction_num = 100
#将linspace返回的行向量转为矩阵
x_predictions = np.linspace(x_train.min(), x_train.max(), prediction_num).reshape(prediction_num,1)
y_predictions = lr.predict(data = x_predictions)  # 将x传入进行预测(即x乘θ得到预测的y,θ经过上面的过程计算出)
plt.scatter(x_train, y_train, label='Train data') #训练数据散点图
plt.scatter(x_test, y_test, label='Test data')    #测试数据散点图
plt.plot(x_predictions, y_predictions, 'r', label='Predictions') #预测数据直线
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Prediction y')
plt.legend()
plt.show()
