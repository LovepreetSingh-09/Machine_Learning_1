# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 01:11:13 2019

@author: user
"""
# Gradient Boosted Regression 
# It can be used for regression and classification despite regression in the name
# There is no randomization in it.
# gradient boosting works by building trees in a serial manner, where each tree tries to correct the mistakes of the previous one. 
# learning_rate controls how strongly each tree tries to correct the mistakes of the previous trees.
import graphviz
import mglearn
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
from IPython.display import display
from sklearn.model_selection import train_test_split

# By default, 100 trees of maximum depth 3 and a learning rate of 0.1 are used
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,random_state=0)
gbrt=GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train,y_train)
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train))) 
# Accuracy on training set: 1.000 and Accuracy on test set: 0.958
# Hence there is overfitting. So, to reduce it, limit maximum depth and lower the learning rate
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test))) 

gbrt=GradientBoostingClassifier(random_state=0,max_depth=1)
gbrt.fit(X_train,y_train)
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train))) 
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test))) 
# Accuracy on training set: 0.991 and Accuracy on test set: 0.972
# So, now overfitting reduce and test score is increased
gbrt = GradientBoostingClassifier(random_state=0,learning_rate=0.01) 
gbrt.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train))) #Accuracy on training set: 0.988
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))  # Accuracy on test set: 0.965

gbrt = GradientBoostingClassifier(random_state=0,n_estimators=4,learning_rate=0.01) 
gbrt.fit(X_train, y_train)
# So, n_estimators makes the model complex and increasing its value reduces the accuracy unlike Random Forest
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train))) # Accuracy on training set: 0.627
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test))) # Accuracy on test set: 0.629

gbrt=GradientBoostingClassifier(random_state=0,max_depth=1)
gbrt.fit(X_train,y_train)
print('Feature Importances : ',gbrt.feature_importances_)
# In Gradient Boosting Classifier all the features are not used
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
plot_f_i_cancer(gbrt)
plt.show()
