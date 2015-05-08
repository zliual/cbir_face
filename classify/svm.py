#! /usr/bin/python

import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import Scaler
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold

iris_data = datasets.load_iris()
X = iris_data.data
Y = iris_data.target
print len(X), len(Y), len(X[0])

clf = SVC(kernel='rbf', C=1, gamma=0.01)
clf = clf.fit(X,Y)
y_pred = clf.predict(X)
print y_pred
print Y
#C_range = 10. ** np.arange(-2,9)
#gamma_range = 10. ** np.arange(-5,4)
#param_grid = dict(gamma = gamma_range, C=C_range)

#grid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(y=Y,k=5))
#grid.fit(X,Y)

#score_dict = grid.grid_scores_


