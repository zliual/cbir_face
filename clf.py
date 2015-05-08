#! /usr/bin/env python

'''
import cv2
import numpy as np
import scipy.ndimage as nd
from skimage.feature import local_binary_pattern as lbp
'''

import numpy as np
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit


def ft_load(path):
	features = []
	return features

def save_clf(path):
	pass

def train(X, Y, clf, param):
	return clf


