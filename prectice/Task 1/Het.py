from sklearn import datasets  
                     
from sklearn.neighbors import KNeighborsClassifier 
#import numpy as np 
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt 

iris=datasets.load_iris() 

iris_x=iris.data   
 
iris_y=iris.target  


#iris_x_train, iris_x_test, iris_y_train, iris_y_test = train_test_split(iris_x, iris_y, test_size = 0.3)


knn_clf = KNeighborsClassifier()
k_range = range(1, 11)
weight_range=['uniform','distance']
metric_range=['euclidean','manhattan','chebyshev','minkowski']#,'wminkowski','seuclidean','mahalanobis'
param_grid={
        'weights':weight_range,
        'n_neighbors':[i for i in range(1, 11)],
        'metric':metric_range
}
        


grid_search = GridSearchCV(knn_clf, param_grid, scoring = 'f1_macro', n_jobs = 1, cv = 5)
grid_search.fit(iris_x,iris_y)
print("best parameters:")
print(grid_search.best_estimator_)
print("score:")
print(grid_search.best_score_)

k_range = range(1, 11)
k_scores = []
for k in k_range:

    knn = KNeighborsClassifier(n_neighbors = k, weights = grid_search.best_params_['weights'],
                               metric = grid_search.best_params_['metric'])
    scores = cross_val_score(knn, iris_x, iris_y, cv = 5, n_jobs = 1,scoring = 'f1_macro')
    k_scores.append(scores.mean())

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()



