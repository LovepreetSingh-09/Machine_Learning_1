# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 00:18:50 2019

@author: user
"""
# Navie Bayes Classifier

import mglearn
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split

# There are three kinds of naive Bayes classifiers implemented in scikitlearn: GaussianNB, BernoulliNB, and MultinomialNB. 
# GaussianNB can be applied to any continuous data
# BernoulliNB assumes binary data 
# MultinomialNB assumes count data (that is, that each feature represents an integer count of something, like how often a word appears in a sentence). 
# BernoulliNB and MultinomialNB are mostly used in text data classification.
# Here We used the BernouliNB because The BernoulliNB classifier counts how often every feature of each class is not zero. 
X=np.array([[1,0,1,0],[1,0,0,0],[0,1,0,1],[1,1,0,0]])
y=np.array([1,0,1,0]) # It acts as the ouput of X here, 1 is the output of 0 and 2 row
counts={}
print(X,'\n',y,'\n',X.sum(axis=0)) # axis=0 means sum of all the rows of a particular columns
print(np.unique(y)) # Output=[0,1]
for label in np.unique(y):
    counts[label]=X[y==label].sum(axis=0) 
    print(X[y==label])# Firstly, It will print 0 and 2nd row which belongs to 0 and then prints 1st and 3rd row which belongs to 1 
print('Total Features Counts: ',counts)
