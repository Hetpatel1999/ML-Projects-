# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:17:55 2019

@author: Het
"""
#import all librarys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm

#chips.csv is a data set 
chips = pd.read_csv('chips.csv')
X = [chips['x'], chips['y']]
Y = chips['class']

X=np.array(X).T
Y=np.array(Y).T
#it will split value ,user values for test set and train set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#set perameter to dictionary.
models = [{'svc':svm.SVC(kernel='rbf', C=100, gamma='auto'), 'kernel':'rbf'},
          {'svc':svm.SVC(kernel='linear', C=100), 'kernel':'linear'},
          {'svc':svm.SVC(kernel='poly', C=100, gamma='auto', degree=2), 'kernel':'poly'},
          {'svc':svm.SVC(kernel='sigmoid', C=1000, gamma='auto'), 'kernel':'sigmoid'}]
scores = []
#for loop is, set all dictionary in loop.
for model in models:
    clf = model['svc']
    clf.fit(X_train, y_train)
#for add point in graph.
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    
#for set pools
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
#for ploting in graph
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    
# set graph as 3D
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1])
    
#for get best score in output
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k')
    score=clf.score(X_test, y_test)
    print(model['kernel'], "score:", score)
    scores.append(score)
    plt.title(model['kernel'])
    plt.show()
    
print("The best score is:",max(scores))