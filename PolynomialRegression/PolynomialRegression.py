import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge  # 岭回归
from sklearn.linear_model import Lasso  # lasso模型
import matplotlib.pyplot as plt

# 多项式回归

# 数据构建和模型训练
m = 1000
X = 6 * np.random.rand(m, 1) - 3  # 取值-3到3
y = 0.5 * X ** 2 + X + np.random.randn(m, 1)  # y=0.5x^2+x+正态分布的抖动
poly_features = PolynomialFeatures(degree=2, include_bias=False)  # degree为最高幂次,过高有过拟合风险,bias为内置偏置值
X_poly = poly_features.fit_transform(X, y)  # 先fit得到特征变换的规则,再transform将一个特征值变为两个特征值x和x^2
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)  # 对x^2和x进行线性回归拟合
print(lin_reg.coef_)  # 参数分别为0.5和1

# 预测
X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new_predict = lin_reg.predict(X_new_poly)

# 绘制拟合曲线
plt.plot(X, y, 'b.')
plt.plot(X_new, y_new_predict, 'r--', label='prediction')
plt.axis([-3, 3, -5, 10])
plt.legend()
plt.show()
