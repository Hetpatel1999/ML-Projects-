import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MiniBatchKMeans
from sklearn import datasets
from sklearn.metrics import confusion_matrix, adjusted_rand_score
import pandas as pd

iris = datasets.load_iris()
X = iris.data[:, :4]
print(X.shape)
plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c="yellow", marker='o', label='see')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)

plt.figure(2)
estimator = KMeans(n_clusters=3)
estimator.fit(X)
label_pred = estimator.labels_

x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], c="yellow", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
plt.title('KMEANS')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)


#score for geting
print('Kmeans:', adjusted_rand_score(iris.target, estimator.labels_))

irisdata = iris.data

clustering = AgglomerativeClustering(linkage='ward', n_clusters=3)

res = clustering.fit(irisdata)

print('Agnes:', adjusted_rand_score(iris.target, clustering.labels_))

plt.figure(3)
d0 = irisdata[clustering.labels_ == 0]
plt.scatter(d0[:, 0], d0[:, 1], c="yellow", marker='o', label='label0')
d1 = irisdata[clustering.labels_ == 1]
plt.scatter(d1[:, 0], d1[:, 1], c="green", marker='*', label='label1')
d2 = irisdata[clustering.labels_ == 2]
plt.scatter(d2[:, 0], d2[:, 1], c="blue", marker='+', label='label2')
plt.xlabel("Sepal.Length")
plt.ylabel("Sepal.Width")
plt.title("AGNES")
plt.legend(loc=2)


X = iris.data[:, :4]
plt.figure(4)
dbscan = DBSCAN(eps=0.4, min_samples=9)
dbscan.fit(X)
label_pred = dbscan.labels_

print('Dbscan:', adjusted_rand_score(iris.target, dbscan.labels_))
#
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], c="yellow", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title('DBSCAN')
plt.legend(loc=2)

X = iris.data[:, :4]  #
plt.figure(5)
estimator = MiniBatchKMeans(n_clusters=3)  #
estimator.fit(X)  #
label_pred = estimator.labels_  #
#
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], c="yellow", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
plt.title('MiniBatchKMeans')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
print('MiniBatchKMeans:', adjusted_rand_score(iris.target, estimator.labels_))

plt.show()

#print(ars)