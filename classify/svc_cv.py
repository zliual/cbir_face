#! /usr/bin/python

import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

iris_data = datasets.load_iris()
X = iris_data.data
Y = iris_data.target

C_range = np.logspace(-2, 10, 13)
g_range = np.logspace(-5, 5, 11)
ker_range = ['rbf', 'poly']
param_dict = dict(gamma=g_range, C=C_range, kernel=ker_range)
cv = StratifiedShuffleSplit(Y, n_iter=5, test_size=0.2, random_state=42) 
clf = SVC()
grid = GridSearchCV(clf, param_grid=param_dict, cv=cv)
clf = grid.fit(X,Y).best_estimator_ 
y_pred = clf.predict(X)
print y_pred
print Y

