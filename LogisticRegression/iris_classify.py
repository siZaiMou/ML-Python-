import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

# 数据处理
iris = datasets.load_iris()
print(iris.keys())
X_train = iris['data'][:120,:]
y_train = iris['target'][:120]
# 使用sklearn的逻辑回归进行预测
lgr = LogisticRegression()
lgr.fit(X_train, y_train)
X_test = iris['data'][120:,:]
y_test = iris['target'][120:]
y_pred = lgr.predict(X_test)
precision = np.sum(y_pred==y_test)/30 * 100
print(precision)