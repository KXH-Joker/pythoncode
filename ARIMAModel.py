# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 10:21:16 2017

@author: joker
"""
#             自相关     偏自相关
#    AR       拖尾        截尾
#    MA       截尾        拖尾
#   ARMA      拖尾        拖尾
#拖尾与截尾的判断
#拖尾是指尾部没有收敛 仍然变动较大 截尾是尾部收敛逐渐收敛为0 变动较小或者没有
#ARIMA  和 ARMA都是对平稳数据建模 ARIMA中间加了一个差分的过程 其余的都一样

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf   #做自相关图的
from statsmodels.graphics.tsaplots import plot_pacf  #做偏自相关图的
from statsmodels.tsa.stattools import adfuller as ADF  #平稳性检验
from statsmodels.stats.diagnostic import acorr_ljungbox  #白噪声检验
from statsmodels.tsa.arima_model import ARIMA   #建立ARIMA模型
 
forecastnum = 5

data = pd.read_excel('C:/Users/joker/Desktop/arima_data.xls',index_col='日期')

plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

data.plot()
plt.show()

#自相关图
#plot_acf(data).show()

#平稳性检测
print('===========平稳性检测===========')
#print('原始序列的adf检验结果为：',ADF(data['销量']))
#上面的返回值依次是：adf p-value usedlag nobs critical 
#values icbest regresults resstore  这一个重点关注一些p值 ，p值大于0.05就是非平稳序列
#差分
print('===========差分===========')
D_data = data.diff().dropna()
D_data.columns = ['销售差分']
print(D_data)
D_data.plot()
plt.show()

plot_acf(D_data).show()
plot_pacf(D_data).show()
print("差分之后的adf检验结果：",ADF(D_data['销售差分']))

#白噪声检验
print('==========白噪声检验============')
print('差分之后的序列白噪声检验：',acorr_ljungbox(D_data,lags=1)) 
#返回值是统计量和p值 重点关注p值 p值小于0.05为非白噪声序列 所以上面的平稳非白噪声序列

#定阶
data['销量'] = data['销量'].astype(float)
pmax = int(len(D_data)/10)  #一般阶数部超过length/10
qmax = int(len(D_data)/10)
print(pmax,qmax)
bic_matrix = [] #bic矩阵
for p in range(pmax+1):
    tmp = []
    for q in range(qmax+1):
        try:
            tmp.append(ARIMA(data,(p,1,q)).fit().bic)
        except:
            tmp.append(None)
    bic_matrix.append(tmp)


bic_matrix = pd.DataFrame(bic_matrix) #从中找出最小值
print(bic_matrix)

p,q = bic_matrix.stack().idxmin() #使用stack展平 然后找出最小值位置 
print('bic最小的p和最小的q为: %s \ %s'  %(p,q))
model = ARIMA(data,(p,1,q)).fit() #建立模型arima（0，1，1）
result = model.summary2()
print(result)

test = model.forecast(5) #给出未来五天的预测 返回预测结果 标准误差 置信区间
print(test)









