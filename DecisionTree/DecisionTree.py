import operator
from math import log



class DecisionTree:
    def __init__(self, dataset, features):
        self.dataset = dataset  # dataset包含特征和标签
        self.features = features  # 特征的名字们
        self.featlabels = []  # 每层递归传递的特征(这棵树依次进行分类的特征)

    def train(self):
        self.tree = self.createTree(self.dataset, self.features)  # 建立决策树
        return self.featlabels

    def createTree(self, dataset, features):
        classList = [example[-1] for example in dataset]  # 当前数据集的最后一列(y值)
        if classList.count(classList[0]) == len(classList):  # 这一列全部相同,熵值为0,作为叶子节点
            return classList[0]
        if len(dataset[0]) == 1:  # 当前数据集只有一列,数据集到这一层已遍历完,所有特征都计算过(第一行长度为1)
            return self.majorityCnt(classList)  # 计算当前节点中哪个类别比较多作为叶节点
        # 以上两个if为递归停止条件
        bestFeatIndex = self.chooseBestFeatureToSplit(dataset)  # 在当前数据集中选出最优特征的列号(信息增益)
        bestFeat = features[bestFeatIndex]
        self.featlabels.append(bestFeat)
        myTree = {bestFeat: {}}  # 树的结构为嵌套字典,最外层key代表根节点
        del features[bestFeatIndex]  # 在特征名中将选出的最优特征删掉
        featValues = set([example[bestFeatIndex] for example in dataset])  # 当前选出最优特征的值并去重
        for value in featValues:  # 当前特征有几个取值就分几叉
            # 递归调用,生成以当前选出的特征为根节点不同取值情况下的子树,splitDataSet根据当前层最优特征的取值分割dataset
            myTree[bestFeat][value] = self.createTree(self.splitDataSet(dataset, bestFeatIndex, value), features)
        return myTree

    def majorityCnt(self, classList):
        classCount = {}  # 字典,计算各个值的出现次数
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote] = 0;
            classCount[vote] += 1
        sorted = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 得到出现次数最多的key
        return classCount[0][0]

    # 计算最好特征
    def chooseBestFeatureToSplit(self, dataset):
        numFeatures = len(dataset[0])-1  # 特征个数
        # 计算这一层的基础熵值(在这层不选定任何其它特征)
        baseEntropy = self.calcShannonEnt(dataset)
        bestInfoGain = 0  # 最好的信息增益率
        bestFeatureIndex = -1
        for featureIndex in range(numFeatures):  # 遍历当前每个特征对应的列
            featValues = set([example[featureIndex] for example in dataset])
            newEntropy = 0
            for value in featValues:  # 计算当前特征的加权平均熵
                subDataSet = self.splitDataSet(dataset, featureIndex, value)  # 根据特征及特征取值切分数据集
                prob = len(subDataSet) / float(len(dataset))  # 当前特征取value的概率
                newEntropy += prob * self.calcShannonEnt(subDataSet)  # 加权平均熵 pi*E(a,y)
            infoGain = baseEntropy - newEntropy
            # 更新最优特征
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeatureIndex = featureIndex
        return bestFeatureIndex

    # 计算不同切分数据集下的熵值(y的熵值)
    def calcShannonEnt(self,dataset):
        numExamples = len(dataset)
        labelCounts = {}  # 每个标签出现的次数
        for featVec in dataset:
            currentLabel = featVec[-1]  # 当前一行的最后一列(当前样本的标签)
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        shannonEnt = 0
        for key in labelCounts:
            prop = float(labelCounts[key]) / numExamples  # 计算每个key出现的概率值 ∑pi*logpi
            shannonEnt -= prop * log(prop, 2)  # 取负号
        return shannonEnt

    # 根据特征及特征取值切分数据集(返回特征featureIndex取值为value的行,并删除当前的特征列)
    def splitDataSet(self,dataset, featureIndex, value):
        retDataSet = []
        for featVec in dataset:
            if featVec[featureIndex] == value:
                splitVec = featVec[:featureIndex]
                splitVec.extend(featVec[featureIndex + 1:])
                retDataSet.append(splitVec)
        return retDataSet
