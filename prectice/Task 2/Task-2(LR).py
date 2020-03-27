# -*- coding: utf-8 -*-
"""
Created on Wed Jul 9 15:23:33 2019
-
@author: Het
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error
from math import sqrt

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


prices_data0 = pd.read_csv('prices.csv')  


#print(prices_data.columns)
X=[prices_data0['area'],prices_data0['rooms']]
y=prices_data0['price']

Xdata=np.array(X)
ydata=np.array(y)
Xdata_mine=np.r_[Xdata,np.ones([1,len(Xdata[0])])].T

weightX=np.array([0.2, 1, 0.5],)


epsilon = 0.001
 

alpha = 1e-7
diff = [-1, 1]
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


    
    
    # the error function 
    Y_pred = np.dot(Xdata_mine,weightX)
    error1=sqrt(mean_squared_error(y, Y_pred))

    error.append(error1);
    if cnt > 1:
        if abs(error[cnt-1] - error[cnt-2]) < epsilon:
            break
        else:
            pass

 
print('************************no pre-processed data********************')

plt.plot(list(range(0,cnt)), error)
plt.xlabel('the number of iterations')
plt.ylabel('the error function ')
#plt.show()


print('min error of mine:',min(error))


linreg = LinearRegression()
linreg.fit(Xdata.T, ydata)


skerror=linreg.predict( Xdata.T)
print('error of sklearn:',sqrt(mean_squared_error(ydata, skerror)))

#=========================MinMaxScaler
print('******************** min-max scaled data*****************')
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

alpha = 1e-7
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

 
plt.plot(list(range(0,cnt)), error)
plt.xlabel('the number of iterations')
plt.ylabel('the error function ')
#plt.show()

print('min error of mine:',min(error))


linreg = LinearRegression()
linreg.fit(Xdata2.T, ydata)

skerror=linreg.predict( Xdata2.T)
print('error of sklearn:',sqrt(mean_squared_error(ydata, skerror)))


   
#=========================StandardScaler
print('*****************standardized scaled data*********************')
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


alpha = 1e-7
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

 

plt.plot(list(range(0,cnt)), error)
plt.xlabel('the number of iterations')
plt.ylabel('the error function ')
plt.show()

print('min error of mine:',min(error))


linreg3 = LinearRegression()
linreg3.fit(Xdata3.T, ydata)
#print (linreg.intercept_)
#print ('anser of sklearn:',linreg.coef_)
skerror3=linreg3.predict( Xdata3.T)
print('error of sklearn:',sqrt(mean_squared_error(ydata, skerror3)))

