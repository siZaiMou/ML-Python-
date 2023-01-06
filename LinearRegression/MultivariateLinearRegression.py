import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
# plotly.offline.init_notebook_mode()
from LinearRegression import LinearRegression

# 多特征值回归预测


# 读入数据
data = pd.read_csv('data/world-happiness-report-2017.csv')
train_data = data.sample(frac=0.8)  # 训练数据取80%的样本
test_data = data.drop(train_data.index)  # 原始样本删掉训练样本作为测试数据
input_param_name1 = 'Economy..GDP.per.Capita.'  # 根据多个指标预测"Happiness.Score"
input_param_name2 = 'Freedom'
input_param_name3 = 'Family'
input_param_name4 = 'Health..Life.Expectancy.'
input_param_name5 = 'Generosity'
input_param_name6 = 'Trust..Government.Corruption.'
input_param_name7 = 'Dystopia.Residual'
output_param_name = 'Happiness.Score'
x_train = train_data[
    [input_param_name1, input_param_name2, input_param_name3, input_param_name4, input_param_name5, input_param_name6,
     input_param_name7]].values  # 两个特征值
y_train = train_data[[output_param_name]].values  # labels
x_test = test_data[[input_param_name1, input_param_name2]].values
y_test = test_data[[output_param_name]].values

# 训练
num_iterations = 500
learning_rate = 0.01
lr = LinearRegression(x_train, y_train)
(theta, cost_history) = lr.train(learning_rate, num_iterations)
print(theta)
print(cost_history[0] ,' ' , cost_history[-1])
# 预测
