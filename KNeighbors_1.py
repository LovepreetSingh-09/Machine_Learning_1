# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 00:15:23 2019

@author: user
"""

#import mglearn
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
# Generate dataset
X,y=mglearn.datasets.make_forge() # It is a binary classification based dataset
print(X,'\n',y,'\nThe shape of x is : ',X.shape)
# Plot dataset The plot is X(column 1) v/s x(column 2) and y represent the class in 0 and 1
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.legend(['Class 0','Class 1'],loc='best')
plt.xlabel('First Feature')
plt.ylabel('Second Feature')
plt.show()

X,y=mglearn.datasets.make_wave(n_samples=40) # It is a regression based dataset
print(X,'\n',y,'\nThe shape of x is : ',X.shape,'\nThe shape of y is : ',y.shape)
plt.plot(X,y,'go')
plt.ylim(-3,3)
plt.xlabel('Feature')
plt.ylabel('Target')
plt.show()

# Datasets that are included in scikit-learn are usually stored as Bunch objects. they behave like dictionaries, with the added benefit that you can access values using a dot (as in bunch.key instead of bunch['key']).
from sklearn.datasets import load_breast_cancer # It is a binary-classification based dataset because of two types of cancer
dataset=load_breast_cancer()
print('The Keys are : ',dataset.keys())
print('The shape of data is:',dataset.data.shape)
uniqs,counts=np.unique(dataset.target,return_counts=True)
# Gives the count of an element in the key
print(np.bincount(dataset.target))
# makes a dictionary of target name with their target count
print("Sample counts per class:\n{}".format({n: v for n, v in zip(dataset.target_names, np.bincount(dataset.target))}))
print("Feature names:\n{}".format(dataset.feature_names))
print("Feature names:\n{}".format(dataset.feature_names.shape))
# Simple Data having size (506,13) It has 13 features
from sklearn.datasets import load_boston
boston = load_boston()
print('The keys are:',boston.keys())
print("Data shape: {}".format(boston.data.shape)) 
print("Data shape: {}".format(boston.feature_names)) 
# Extended dataset and having size(506,104) having 104 new features.These new features represent all possible interactions between two different original features, as well as the square of each original feature. 
# 13= orifinal features,13=square of original features 78=12+11...+1 is the combination of 2 features
X, y = mglearn.datasets.load_extended_boston() 
print("X.shape: {}".format(X.shape)) 
#print("X: {}".format(X))
# General plot to represent how k no. of neighbors work
mglearn.plots.plot_knn_classification(n_neighbors=3)

from sklearn.model_selection import train_test_split 
X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.neighbors import KNeighborsClassifier 
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
print("Test set predictions: {}".format(clf.predict(X_test)))
print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test))) 
fig,axes=plt.subplots(1,3,figsize=(10,3))
# the fit method returns the object self, so we can instantiate and fit in one line
for n_neighbors,ax in zip([1,3,9],axes):
    clf=KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y)
    # fill=True fills the colour in the distinguished region and eps determines the whole area of plot to be coloured So it should be more than 0.4
    mglearn.plots.plot_2d_separator(clf,X,fill=True,eps=0.6,ax=ax,alpha=0.5)
    mglearn.discrete_scatter(X[:,0],X[:,1],y,ax=ax)
    ax.set_title('{} neighbors'.format(n_neighbors))
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
axes[1].legend(loc='best')