# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 00:39:03 2019

@author: user
"""
# Descision Tree
#import graphviz
import mglearn
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split

mglearn.plots.plot_animal_tree()

from sklearn.tree import DescisionTreeClassifier
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=42)
dtc=DescisionTreeClassifier().fit(X_train,y_train)
print("Training set score: {:.3f}".format(dtc.score(X_train, y_train)))
print("Test set score: {:.3f}".format(dtc.score(X_test, y_test)))
