# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:51:18 2018

@author: joker
"""

'''
    这是一个聚类的例子
    利用谱系图来初步确定分类的类别数
    再利用聚类算法进行分类  
    
    这个可以和kmeans聚类算法结合一波
'''
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage,dendrogram
from sklearn.cluster import AgglomerativeClustering  

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

date = pd.read_excel("C:/Users/joker/Desktop/date/business_circle.xls", index_col = u'基站编号')

'''
    数据标准化 ，因为数据量级差别较大为了减少数据带来的影响 对数据进行标准化
    (x - mean)/std     范围[-1,1]
    (x - min )/ (max-min)    范围[0,1]
'''
date = (date - date.min()) / (date.max() - date.min())
date = date.reset_index()

'''
    绘制谱系图
    linkage(数据，使用的算法（又ward ， median，centroid，weighted，average ，complete ，single）
            算法的区别是分的层数于细致程度 默认是single
    metric默认是euclidean   不修改)
    Scipy包可以看作是补充numpy的包提供了一些相应的方法和计算 这里利用了其层次聚类的方法绘制谱系图
    dendrogram  Plots the hierarchical clustering as a dendrogram.
    参数很多 可以直接查看源代码
'''
Z = linkage(date , method='ward' , metric='euclidean')
p = dendrogram(Z,0) 
plt.show()

k = int(input('k:    '))
'''
    利用sklearn包的层次聚类进行分析
'''

model = AgglomerativeClustering(n_clusters=k , linkage= 'ward')
model.fit(date)

#输出详细的 数据
r = pd.concat([date,pd.Series(model.labels_ , index = date.index)] , axis =1)
r.columns = list(date.columns) + ["类别"]
#print(r[r['类别'] == 0])


style = ['ro-','g*-','b+-']
xlabels = ['工作日人均停留时间','凌晨人均停留时间','周末人均停留时间','日均人流量']

for i in range(k):
    print(i)
    plt.figure()
    tmp = r[r['类别'] == i].iloc[:,:4]
    print(tmp.head())
    print("+++++++++++++++++++")
    for j in range(len(tmp)):
        plt.plot(range(1,5),tmp.iloc[j],style[i])
    
    plt.xticks(range(1,5) , xlabels , rotation = 20)
    plt.title(u'商圈类别%s' %(i+1))
    plt.subplots_adjust(bottom = 0.15)
