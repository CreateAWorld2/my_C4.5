from cmath import log
import math
import numpy as np

 
class C45:

    def __init__(self,dataPath,dataName) -> None:

        self.dataFile = open(dataPath,'r')
        self.data = [line.strip().split(',') for line in self.dataFile.readlines()]
        self.data = np.array(self.data,dtype=np.int32)

        self.dataName = open(dataName,'r',encoding='utf-8')
        self.name = self.dataName.readline().split(',')


    def cac_entropy(self,data:np.ndarray):
        '''
        ?????????
        '''
        entropy = 0
        row,col = data.shape
        classData = data[:,-1]

        labelDict = {}
        for currentclass in classData:
            if currentclass not in labelDict.keys():
                labelDict[currentclass] = 0
            labelDict[currentclass]+=1
        
        for key in labelDict:
            tmp = -(labelDict[key]/row)*math.log2(labelDict[key]/row)
            entropy += tmp
        
        return entropy

    
    def splitdataset(self,data:np.ndarray,attrIndex,value):
        '''
        分割数据集
        输入：数据集，需要分割的属性列，分割值
        返回：np.ndarray，分割的subcase

        '''
        subsetData = []
        dataList = data.tolist()

        for valueVec in dataList:
            #选出与分割属性列中与分割值相等的case
            if(valueVec[attrIndex] == value):
                subsetTmp = valueVec[:attrIndex]
                subsetTmp.extend(valueVec[attrIndex+1:])
                subsetData.append(subsetTmp)
        return np.array(subsetData)

    
    def choseBestAttribute(self,data:np.ndarray):
        '''
        寻找最适合作test的attribute
        '''
        classData = data[:,-1]
        baseInfo = self.cac_entropy(data)
        bestAttr = -1
        dataLen = data.shape[1]-1
        maxGain = 0
        for attr in range(dataLen):
            #遍历每个属性，选择最大的gain ratio
            uniqueValue = set(data[:,attr])
            infoXT = 0  #测试X分割原始集合T的info
            for uni in uniqueValue:
                subsetData = self.splitdataset(data,attr,uni)
                infoSubsetData = self.cac_entropy(subsetData)
                propotion = (len(subsetData)/float(len(classData)))
                infoXT += propotion*infoSubsetData
            
            gainXT = baseInfo - infoXT
            print(u"第%d个信息增益为：%.3f" %(attr,gainXT))
            if(gainXT>maxGain):
                maxGain = gainXT
                bestAttr = attr

        return bestAttr

    def majarityAttr(self,classList):
        labelDict = {}
        count= 0
        for currentclass in classList:
            if currentclass not in labelDict.keys():
                labelDict[currentclass] = 0
            labelDict[currentclass]+=1

        for value in labelDict.values():
            if(value>count):
                count = value
        return count


    
    def CreateTree(self,data:np.ndarray,labels):
        classList = data[:,-1].tolist()
        if(classList.count(classList[0]) == len(classList)):
            #类别只有一个类
            print("该属性最终类别为:"+str(classList[0]))
            return classList[0]
        if(len(data[0,:]) == 1):
            #数据集只有一个值
            return self.majarityAttr(classList)

        bestAttr = self.choseBestAttribute(data)
        bestLabel = labels[bestAttr]
        print(u"当前最优分类属性为:" + (bestLabel))
        C45Tree = {bestLabel:{}}
        del (labels[bestAttr])

        attrValue = data[:,bestAttr]
        uniqueValue = set(attrValue)

        for value in uniqueValue:
            subLabel = labels[:]
            C45Tree[bestLabel][str(value)] = self.CreateTree(self.splitdataset(data,bestAttr,value),subLabel)

        return C45Tree


if __name__ == "__main__":
    c45 = C45('dataset.txt','name.txt')
    
    #测试数据
    data = np.array([[0,0,0,0,0],
                    [0,0,0,1,0],
                    [0,1,0,1,1],
                    [0,1,1,0,1],
                    [0,0,0,0,0],
                    [1,0,0,0,0],
                    [1,0,0,1,0],
                    [1,1,1,1,1],
                    [1,0,1,2,1],
                    [1,0,1,2,1],
                    [2,0,1,2,1],
                    [2,0,1,1,1],
                    [2,1,0,1,1],
                    [2,1,0,2,1],
                    [2,0,0,0,0],
                    [2,0,0,2,0]])
    labels = ['年龄段','有工作','有房子','信贷情况','类别']
    
    #print(c45.splitdataset(data,1,0))
    #print(c45.cac_entropy(c45.splitdataset(data,1,0)))
    #print(c45.name)
    #print(c45.choseBestAttribute(data))
    print(c45.CreateTree(data,labels))

