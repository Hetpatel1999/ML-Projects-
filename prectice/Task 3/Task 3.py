# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:53:43 2019

@author: Het
"""

import os
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def convert_to_sparse_matrix(docs):
	indptr = [0]
	indices = []
	data = []
	vocabulary = {}
	for d in docs:
		for term in d:
			index = vocabulary.setdefault(term, len(vocabulary))
			indices.append(index)
			data.append(1)
		indptr.append(len(indices))
	return csr_matrix((data, indices, indptr), dtype=int)


def get_class(spam_prob):
	if spam_prob < 0.99999:
		return "legit"
	else:
		return "spam"


data_dir = 'messages'
messages = []
y = []
for filename in os.listdir(data_dir):
	with open (os.path.join(data_dir, filename)) as file:
		subject = file.readline()
		skip = file.readline()
		content = file.readline()
		messages.append(content.split())
	if "legit" in filename:
		y.append("legit")
	else:
		y.append("spam")

X = convert_to_sparse_matrix(messages)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = MultinomialNB(fit_prior=False, class_prior = [0.999,0.000000001])
clf.fit(X_train, y_train)
y_prob = clf.predict_proba(X_test)

print(confusion_matrix(y_test, clf.predict(X_test)))
fpr, tpr, threshold = roc_curve(y_test, y_prob[:,0], pos_label = 'legit')#legit to spam

plt.plot(fpr, tpr,'red')
plt.show()









