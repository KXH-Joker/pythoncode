# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:29:18 2017

@author: joker
"""
#apriori的实现类

from __future__ import print_function
import pandas as pd

#自定义连接函数
def connect_string(x,ms):
    x = list(map(lambda i : sorted(i.split(ms)),x))
    l = len(x[0])
    r = []
    for i in range(len(x)):
        for j in range(i,len(x)):
            if x[i][:l-1] == x[j][:l-1] and x[i][l-1] != x[j][l-1]:
                r.append(x[i][:l-1] + sorted(x[j][l-1],x[i][l-1]))
    return r


#寻找关联规则的实现方法
#d ==>数据   support ==> 最小支持度 confidence ==>最小置信度 ms ==>连接方式这个随意
def find_rule( d , support , confidence , ms = '--'):
    result = pd.DataFrame(index=['support','confidence'])
    support_series = 1.0 * d.sum() / len(d)
    column = list(support_series[support_series > support].index)   #筛选出大于我们设定最小支持度的行
    k=0
    
    while(len(column) > 1):
        k = k+1
        
        print('正在进行第%s次搜索' %k)
        column = connect_string(column,ms)
        print('数目：%s    ' %len(column))
        sf = lambda i : d[i].prod(axis = 1 ,numeric_only = True)
        
        #创建连接数据，耗时耗内存 相对较大，数据较大考虑并行运算优化
        d_2 = pd.DataFrame(list(map(sf,column)),index = [ms.join(i) for i in column]).T
        
        support_series_2 = 1.0 * d_2[[ms.join(i) for i in column]].sum() / len(d)
        
        column = list(support_series_2[support_series_2 > support].index)
        support_series = support_series.append(support_series_2)
        
        column2 = []
        
        for i in column:
            i = i.split(ms)
            for j in range(len(i)):
                column2.append(i[:j]+i[j+1:]+i[j:j+1])
        
        confidence_series = pd.Series(index=[ms.join(i) for i in column2])
        
        for i in column2:
            confidence_series[ms.join(i)] = support_series[ms.join(sorted(i))]/support_series[ms.join(i[:len(i)-1])]
            
        for i in confidence_series[confidence_series > confidence].index:
            result[i] = 0.0
            result[i]['confidence'] = confidence_series[i]
            result[i]['support'] = support_series[ms.join(sorted(i.split(ms)))]
            
        result = result.T.sort(['confidence','support'],ascending = False)
        
        print('结果为：')
        print(result)
        
        return result
        

