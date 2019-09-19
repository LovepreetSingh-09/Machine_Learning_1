# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 01:20:34 2019

@author: user
"""
# Random Forests

# Ensembles are methods that combine multiple machine learning models to create more powerful models. 
# There are two ensemble models that are effective on a wide range of datasets for classification and regression, both of which use decision trees as their building blocks: 
# 1.) random forests 
# 2.) gradient boosted decision trees.

import graphviz
import mglearn
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
from IPython.display import display
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
X,y=make_moons(n_samples=100,noise=0.25,random_state=3)
X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,random_state=42)
# Bootstrap sample - uses 2/3 data points
# n_estimators define the no. of randomly formed decision tree from the dataset 
# If we set max_features to n_features, that means that each split can look at all features in the dataset, and no randomness will be injected.
# If we set max_features to 1, that means that the splits have no choice at all on which feature to test, and can only search over different thresholds for the feature that was selected randomly. 
# Therefore, a high max_features means that the trees in the random forest will be quite similar, and they will be able to fit the data easily, using the most distinctive features. A low max_features means that the trees in the random forest will be quite different, and that each tree might need to be very deep in order to fit the data well.
# Random Forests require more memory and slower to train and test than linear models So, it takes more time
forest=RandomForestClassifier(n_estimators=5,random_state=2)
forest.fit(X_train,y_train)
fig,axes=plt.subplots(2,3,figsize=(20,10))
# Built Trees are stored in forest.estimators_
for i,(ax,tree) in enumerate(zip(axes.ravel(),forest.estimators_)):
    ax.set_title('Tree : {}'.format(i))
    # Below command makes boundaries as well as scattering points
    mglearn.plots.plot_tree_partition(X_train,y_train,tree,ax=ax)
# For the resulted random forest tree plot made from all others random decision tree plots
mglearn.plots.plot_2d_separator(forest,X_train,ax=axes[-1,-1],fill=True,alpha=0.4)
axes[-1,-1].set_title('Random Forest')
# For scattering points on the random forest tree
mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
# OR we can use the following line from testing part
# mglearn.discrete_scatter(X_test[:,0],X_test[:,1],y_test)
plt.show()

X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,random_state=0)
forest=RandomForestClassifier(n_estimators=100,random_state=0)
forest.fit(X_train,y_train)
# Accuracy on training set: 1.000 and Accuracy on test set: 0.972
print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train))) 
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))
# n_jobs= the cores of system to be used in the model
print('Feature Importances : ',forest.feature_importances_)
# In Random Forest almost all the features are used
def plot_f_i_cancer(model):
    n_features=cancer.data.shape[1]
    plt.barh(range(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features),cancer.feature_names)
    # np.arange(n_features) means the list 0 to 29 representing each feature by the next term in the bracket
    print(np.arange(n_features))
    plt.xlabel('Features Importance')
    plt.ylabel('Features')
    plt.ylim(-1,n_features)  
    plt.show()
plot_f_i_cancer(forest)
plt.show()





