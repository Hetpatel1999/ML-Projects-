# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 00:27:46 2019

@author: Het
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error
from math import sqrt

import matplotlib.pyplot as plt  # 绘图库
import warnings
warnings.filterwarnings("ignore")


prices_data0 = pd.read_csv('prices.csv')  
# 读取训练数据
#print(prices_data.columns)
X=[prices_data0['area'],prices_data0['rooms']]
y=prices_data0['price']

Xdata=np.array(X)
ydata=np.array(y)
Xdata_mine=np.r_[Xdata,np.ones([1,len(Xdata[0])])].T

weightX=np.array([1.0, 1.0, 1.0],)
#Y_pred=[]

#Y_pred=np.dot(Xdata,weightX)
#Y_error=sqrt(mean_squared_error(y, Y_pred))
#print("Root mean square error:",sqrt(mean_squared_error(y, Y_pred)))

# 迭代阀值，当两次迭代损失函数之差小于该阀值时停止迭代
epsilon = 0.001
 
# 学习率
alpha = 1e-9
diff = [0, 0]
max_itor = 1000
error = []

cnt = 0
m = len(Xdata)
 
 
while True:
    cnt += 1
 
    # iterations
   
    for i in range(m):
        #  y = weightX[0]* x[0] +  weightX[1] * x[1] +weightX[2] 
        # residual 
        diff[0] = ( weightX[0] * Xdata_mine[i][0] + weightX[1] * Xdata_mine[i][1] +weightX[2]) - ydata[i]
 
        # gradient = diff[0] * x[i][j]
        weightX[0] -= alpha * diff[0] * Xdata_mine[i][0]
        weightX[1] -= alpha * diff[0] * Xdata_mine[i][1]
        weightX[2] -= alpha * diff[0] * Xdata_mine[i][2]

    
    
    # the error function 
    Y_pred = np.dot(Xdata_mine,weightX)
    error1=sqrt(mean_squared_error(y, Y_pred))

    error.append(error1);
    if cnt > 1:
        if abs(error[cnt-1] - error[cnt-2]) < epsilon:
            break
        else:
            pass

 
#    print ' theta0 : %f, theta1 : %f, theta2 : %f, error1 : %f' % (theta0, theta1, theta2, error1)
print('=================no pre-processed data=============')
plt.plot(list(range(0,cnt)), error)
plt.xlabel('the number of iterations')
plt.ylabel('the error function ')
plt.show()

#print ('the number of iterations : ', cnt)
#print ('Done: anser of mine :',  weightX)
print('min error of mine:',min(error))


linreg = LinearRegression()
linreg.fit(Xdata.T, ydata)
#print (linreg.intercept_)
#print ('anser of sklearn:',linreg.coef_)


skerror=linreg.predict( Xdata.T)
print('error of sklearn:',sqrt(mean_squared_error(ydata, skerror)))

#plt.plot(Xdata[0], ydata, '.')
#plt.plot(Xdata[0], np.dot(Xdata_mine,weightX),'-',color='r')
#plt.plot(Xdata[0], linreg.coef_[0]*Xdata[0]+linreg.coef_[1]*Xdata[1]+linreg.intercept_, '-',color='g')
#plt.show()



#=========================MinMaxScaler
print('================= min-max scaled data=============')
scaler0 = MinMaxScaler()
scaler0.fit(Xdata[0].reshape(-1, 1))
MinMaxScaler(copy=True, feature_range=(0, 1))


scaler1 = MinMaxScaler()
scaler1.fit(Xdata[1].reshape(-1, 1))
MinMaxScaler(copy=True, feature_range=(0, 1))
Xdata_2_1=scaler0.transform(Xdata[0].reshape(-1, 1))
Xdata_2_2=scaler1.transform(Xdata[1].reshape(-1, 1))
Xdata2=np.c_[Xdata_2_1,Xdata_2_1].T
Xdata_mine2=np.r_[Xdata2,np.ones([1,len(Xdata2[0])])].T

alpha = 1e-3
diff = [0, 0]
max_itor = 1000
error = []

cnt = 0
m = len(Xdata2)
while True:
    cnt += 1
 
    # iterations
   
    for i in range(m):
        #  y = weightX[0]* x[0] +  weightX[1] * x[1] +weightX[2] 
        # residual 
        diff[0] = ( weightX[0] * Xdata_mine2[i][0] + weightX[1] * Xdata_mine2[i][1] +weightX[2]) - ydata[i]
 
        # gradient = diff[0] * x[i][j]
        weightX[0] -= alpha * diff[0] * Xdata_mine2[i][0]
        weightX[1] -= alpha * diff[0] * Xdata_mine2[i][1]
        weightX[2] -= alpha * diff[0] * Xdata_mine2[i][2]

    
    
    # the error function 
    Y_pred=np.dot(Xdata_mine2,weightX)
    error1=sqrt(mean_squared_error(y, Y_pred))

    error.append(error1);
    if cnt>1:
        if abs(error[cnt-1]-error[cnt-2]) < epsilon:
            break
        else:
            pass

 
#    print ' theta0 : %f, theta1 : %f, theta2 : %f, error1 : %f' % (theta0, theta1, theta2, error1)

plt.plot(list(range(0,cnt)), error)
plt.xlabel('the number of iterations')
plt.ylabel('the error function ')
plt.show()

#print ('the number of iterations : ', cnt)
#print ('Done: anser of mine :',  weightX)
print('min error of mine:',min(error))


linreg = LinearRegression()
linreg.fit(Xdata2.T, ydata)
#print (linreg.intercept_)
#print ('anser of sklearn:',linreg.coef_)
skerror=linreg.predict( Xdata2.T)
print('error of sklearn:',sqrt(mean_squared_error(ydata, skerror)))

#plt.plot(Xdata2[0], ydata, '.')
#plt.plot(Xdata2[0], np.dot(Xdata_mine2,weightX),'-',color='r')
#plt.plot(Xdata2[0], linreg.coef_[0]*Xdata2[0]+linreg.coef_[1]*Xdata2[1]+linreg.intercept_, '-',color='g')
#plt.show()


#=========================StandardScaler
print('=================standardized scaled data=============')
scaler0 = StandardScaler()
scaler0.fit(Xdata[0].reshape(-1, 1))
StandardScaler(copy=True, with_mean=True, with_std=True)


scaler1 = StandardScaler()
scaler1.fit(Xdata[1].reshape(-1, 1))
StandardScaler(copy=True, with_mean=True, with_std=True)
Xdata_3_1=scaler0.transform(Xdata[0].reshape(-1, 1))
Xdata_3_2=scaler1.transform(Xdata[1].reshape(-1, 1))
Xdata3=np.c_[Xdata_2_1,Xdata_2_1].T
Xdata_mine3=np.r_[Xdata2,np.ones([1,len(Xdata2[0])])].T


alpha = 1e-2
diff = [0, 0]
max_itor = 1000
error = []

cnt = 0
m = len(Xdata3)
while True:
    cnt += 1
 
    # iterations
   
    for i in range(m):
        #  y = weightX[0]* x[0] +  weightX[1] * x[1] +weightX[2] 
        # residual 
        diff[0] = ( weightX[0] * Xdata_mine3[i][0] + weightX[1] * Xdata_mine3[i][1] +weightX[2]) - ydata[i]
 
        # gradient = diff[0] * x[i][j]
        weightX[0] -= alpha * diff[0] * Xdata_mine3[i][0]
        weightX[1] -= alpha * diff[0] * Xdata_mine3[i][1]
        weightX[2] -= alpha * diff[0] * Xdata_mine3[i][2]

    
    
    # the error function 
    Y_pred=np.dot(Xdata_mine3,weightX)
    error1=sqrt(mean_squared_error(y, Y_pred))

    error.append(error1);
    if cnt>1:
        if abs(error[cnt-1]-error[cnt-2]) < epsilon:
            break
        else:
            pass

 
#    print ' theta0 : %f, theta1 : %f, theta2 : %f, error1 : %f' % (theta0, theta1, theta2, error1)

plt.plot(list(range(0,cnt)), error)
plt.xlabel('the number of iterations')
plt.ylabel('the error function ')
plt.show()

#print ('the number of iterations : ', cnt)
#print ('Done: anser of mine :',  weightX)
print('min error of mine:',min(error))


linreg3 = LinearRegression()
linreg3.fit(Xdata3.T, ydata)
#print (linreg.intercept_)
#print ('anser of sklearn:',linreg.coef_)
skerror3=linreg3.predict( Xdata3.T)
print('error of sklearn:',sqrt(mean_squared_error(ydata, skerror3)))

#plt.plot(Xdata3[0], ydata, '.')
#plt.plot(Xdata3[0], np.dot(Xdata_mine3,weightX),'-',color='r')
#plt.plot(Xdata3[0], linreg.coef_[0]*Xdata3[0]+linreg.coef_[1]*Xdata3[1]+linreg.intercept_, '-',color='g')
#plt.show()
