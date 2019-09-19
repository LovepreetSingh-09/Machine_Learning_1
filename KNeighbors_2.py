# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 00:22:38 2019

@author: user
"""
import mglearn
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)
train_acc=[]
test_acc=[]
neighbors_setting=range(1,11)
for n_neighbor in neighbors_setting:
    clf=KNeighborsClassifier(n_neighbors=n_neighbor)
    clf.fit(X_train,y_train)
    train_acc.append(clf.score(X_train,y_train))
    test_acc.append(clf.score(X_test,y_test))
plt.plot(neighbors_setting,train_acc,'b',label='Training accuracy')
plt.plot(neighbors_setting,test_acc,'r--',label='Test accuracy')
plt.xlabel('Neighbors')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.show()

# Keighbors regression
mglearn.plots.plot_knn_regression(n_neighbors=6)
plt.show()
mglearn.plots.plot_knn_classification(n_neighbors=1)
plt.show()
from sklearn.neighbors import KNeighborsRegressor
X,y=mglearn.datasets.make_wave(n_samples=40)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
print(X_train,'\n',y_train)
reg=KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train,y_train)
print('The Values of prediction of X_test : ',reg.predict(X_test))
print('The Values of y_test : ',y_test)
print('The accuracy of the prediction is {:.4f} '.format(reg.score(X_test,y_test)))
fig,axes=plt.subplots(1,4,figsize=(20,5))
line=np.linspace(-3,3,1000).reshape(-1,1)
#print(line) Makes a 2D array of shape(1000,1)
for n_neighbors,ax in zip([1,3,6,9],axes):
    reg=KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train,y_train)
    ax.plot(line,reg.predict(line),'g')
    # In the first subplot line prediction is along the Training data plots because of just 1 neighbor
    ax.plot(X_train,y_train,'^',c=mglearn.cm3(0),markersize=8) # mglearn.cm3(0) means blue colour and 1 means red colour 
    ax.plot(X_test,y_test,'v',c=mglearn.cm3(1),markersize=8)
    ax.set_title('{} Neighbors\nTrain_score={:.3f} and Test_score={:.3f}'.format(n_neighbors,reg.score(X_train,y_train),reg.score(X_test,y_test)))
    ax.set_xlabel('Feature')
    ax.set_ylabel('Target')
axes[0].legend(['Model Predictions','Training Data/Target','Test Data/Target'],loc='best')
