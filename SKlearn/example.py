import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier  # 梯度下降分类器
from sklearn.model_selection import cross_val_score  # 计算交叉验证的得分
from sklearn.model_selection import StratifiedKFold  # 划分数据集,每份执行不同操作
from sklearn.base import clone  # 用相同的参数构建新的模型(一样)
from sklearn.model_selection import cross_val_predict  # 结果预测
from sklearn.metrics import confusion_matrix  # 混淆矩阵
from sklearn.metrics import precision_score, recall_score  # 计算精度和召回率
from sklearn.metrics import f1_score  # 计算f1综合得分
from sklearn.metrics import precision_recall_curve  # 不同阈值下精度和召回率 精度/召回率曲线
from sklearn.metrics import roc_curve # 绘制roc曲线

#分类问题拿到模型之后应该怎么评估

# 数据划分
mnist = fetch_openml('mnist_784', parser='auto')  # 读入灰度图
X = mnist["data"]  # DataFrame每个像素点
y = mnist["target"]  # 每行标签
# X_train = X[:60000] #前60000个样本作为训练集
X_train = X[:60]  # tttest
# y_train = y[:60000]
y_train = y[:60]  # tttest
# X_test = X[60000:]#60000之后的样本作为测试集
X_test = X[60:100]  # tttest
# y_test = y[60000:]
y_test = y[60:100]  # tttest
shuffle_index = np.random.permutation(60)  # 打乱60000个样本(洗牌),返回一个索引序列,避免排列顺序影响
X_train = X_train.loc[shuffle_index]
y_train = y_train.loc[shuffle_index]

# 训练模型
sgd_clf = SGDClassifier(max_iter=60, random_state=42)  # 最大迭代次数和随机种子(随机操作种子一样)
sgd_clf.fit(X_train, y_train)  # 训练导入的模型
y_predict = sgd_clf.predict(X_test)  # 预测数据

# 模型评估
arr = cross_val_score(sgd_clf, X_train, y_train, cv=3,
                      scoring='accuracy')  # cv将数据进行3次划分和交叉验证,scoring为得分方式'accuracy'为每份数据的准确率
print(arr)

# 交叉划分验证
skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)  # 构造'切分器',3次划分,随机种子为42
# 每次循环将训练集切分为训练集和验证集,所有操作都是在训练集上进行操作,此处test_index为训练集中划分出的验证集索引
for train_index, test_index in skfolds.split(X_train, y_train):
    clone_clf = clone(sgd_clf)  # 构造新模型对切分出来的数据集进行训练
    X_train_folds = X_train.loc[train_index]  # .loc[]取多个行
    y_train_folds = y_train.loc[train_index]
    X_test_folds = X_train.loc[test_index]
    y_test_folds = y_train.loc[test_index]
    clone_clf.fit(X_test_folds, y_test_folds)  # 用训练集中本次切分出的训练数据训练新模型
    y_pred = clone_clf.predict(X_test_folds)  # 用训练集中本次切分出的验证数据进行预测
    n_correct = sum(y_pred == y_test_folds)  # 计算预测正确的个数
    print(n_correct / len(y_pred))  # 求平均值

# 结果预测 将训练集分为三份,其中两份训练,一份验证,重复三次,每次选取的验证集不同,结果为三次验证集的预测值
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)
print(y_train_pred)

# 预测的混淆矩阵n*n
confu_matrix = confusion_matrix(y_train, y_train_pred)  # confu_matrix[i][j]=k代表真实为i类被预测为j类的样本有k个
print(confusion_matrix)

# 计算预测的精度、召回率和f1
print(precision_score(y_train, y_train_pred, average='micro'))
print(recall_score(y_train, y_train_pred, average='micro'))
print(f1_score(y_train, y_train_pred, average='micro'))

# 获得决策分数而非预测结果,可根据决策分数(阈值)来自行决定预测结果(训练数据集内)
y_scores = sgd_clf.decision_function(X_train)

# 不同阈值下的精度和召回率,此处y_scores有10个指标,与例子中预测是否为五的y_train5不同,故例子中y_scores只有一列)
(precisions, recalls, thresholds) = precision_recall_curve(y_train, y_scores[:, 0], pos_label=1)  # 不同阈值下的精度和召回率

#绘制roc曲线
(fpr,tpr,thresholds) = roc_curve(y_train,y_scores[:,0],pos_label=1)