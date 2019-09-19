# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 07:38:13 2019

@author: user
"""

# Unsupervised Learning

import graphviz
import mglearn
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
from IPython.display import display
from sklearn.model_selection import train_test_split

# Types :-
# 1.) Unsupervised Transformations - representation of the data which might be easier for humans or other machine learning algorithms
# 2.) Clustering - partition data into distinct groups of similar items

# Preprocessing and Scaling :-
mglearn.plots.plot_scaling()
plt.show()
# 1.) The StandardScaler in scikit-learn ensures that for each feature the mean is 0 and the variance is 1, bringing all features to the same magnitude. However, this scaling does not ensure any particular minimum and maximum values for the features. 
# 2.) The RobustScaler works similarly to the StandardScaler that it ensures statistical properties for each feature that guarantee that they are on the same scale. The RobustScaler uses the median and quartiles, instead of mean and variance. This makes the RobustScaler ignore data points that are very different from the rest (like measurement errors). These odd data points are also called outliers, and can lead to trouble for other scaling techniques
# The median of a set of numbers is the number x such that half of the numbers are smaller than x and half of the numbers are larger than x. The lower quartile is the number x such that one-fourth of the numbers are smaller than x, and the upper quartile is the number x such that one-fourth of the numbers are larger than x. .
# 3.) The MinMaxScaler, on the other hand, shifts the data such that all features are exactly between 0 and 1. For the two-dimensional dataset this means all of the data is contained within the rectangle created by the x-axis between 0 and 1 and the y-axis between 0 and 1.
# 4.) The Normalizer does a very different kind of rescaling. It scales each data point such that the feature vector has a Euclidean length of 1. In other words, it projects a data point on the circle (or sphere, in the case of higher dimensions) with a radius of 1. This means every data point is scaled by a different number (by the inverse of its length). This normalization is often used when only the direction (or angle) of the data matters, not the length of the feature vector.

# Data Transformtion :-
# The most common motivations of transforming data are visualization, compressing the data, and finding a representation that is more informative for further processing.
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1)
print(X_train.shape) 
print(X_test.shape) 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train) 
# transform data
X_train_scaled = scaler.transform(X_train)
# print dataset properties before and after scaling 
print("transformed shape: {}".format(X_train_scaled.shape))
print("per-feature minimum before scaling:\n {}".format(X_train.min(axis=0))) 
print("per-feature maximum before scaling:\n {}".format(X_train.max(axis=0)))
print("per-feature minimum after scaling:\n {}".format(X_train_scaled.min(axis=0)))
print("per-feature maximum after scaling:\n {}".format(X_train_scaled.max(axis=0))) 
# transform test data
X_test_scaled = scaler.transform(X_test)
# transform method always subtracts the training set minimum and divides by the training set range, which might be different from the minimum and range for the test set.
# print test data properties after scaling
print("per-feature minimum after scaling:\n{}".format(X_test_scaled.min(axis=0))) 
print("per-feature maximum after scaling:\n{}".format(X_test_scaled.max(axis=0))) 

# Scaling Training and Test the same way
from sklearn.datasets import make_blobs
# make synthetic data
X, _ = make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)
# split it into training and test sets 
X_train, X_test = train_test_split(X, random_state=5, test_size=.1)
# plot the training and test sets 
fig, axes = plt.subplots(1, 3, figsize=(13, 4)) 
axes[0].scatter(X_train[:, 0], X_train[:, 1] ,c=mglearn.cm2(0), label="Training set", s=60) 
axes[0].scatter(X_test[:, 0], X_test[:, 1], marker='^',c=mglearn.cm2(1), label="Test set", s=60)
axes[0].legend(loc='upper left') 
axes[0].set_title("Original Data")
# scale the data using MinMaxScaler
scaler = MinMaxScaler() 
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
# visualize the properly scaled data
axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1] ,c=mglearn.cm2(0), label="Training set", s=60) 
axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], marker='^', c=mglearn.cm2(1), label="Test set", s=60) 
axes[1].set_title("Scaled Data")
# rescale the test set separately
# so test set min is 0 and test set max is 1
# DO NOT DO THIS! For illustration purposes only. 
test_scaler = MinMaxScaler() 
test_scaler.fit(X_test) 
X_test_scaled_badly = test_scaler.transform(X_test)
# visualize wrongly scaled data
axes[2].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1] ,c=mglearn.cm2(0), label="training set", s=60) 
axes[2].scatter(X_test_scaled_badly[:, 0], X_test_scaled_badly[:, 1] ,marker='^', c=mglearn.cm2(1), label="test set", s=60) 
axes[2].set_title("Improperly Scaled Data")
for ax in axes:
     ax.set_xlabel("Feature 0")     
     ax.set_ylabel("Feature 1") 
fig.tight_layout()
plt.show()

# fit_transform Method :-
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# calling fit and transform in sequence (using method chaining)
X_scaled = scaler.fit(X_train).transform(X_train)
# same result, but more efficient computation
X_scaled_d = scaler.fit_transform(X_train)

# Effects of Preprcessing:-
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,random_state=0)
svm = SVC(C=100) 
svm.fit(X_train, y_train)
print("Test set accuracy: {:.2f}".format(svm.score(X_test, y_test))) # Test set accuracy: 0.63
# preprocessing using 0-1 scaling 
scaler = MinMaxScaler() 
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
# learning an SVM on the scaled training data 
svm.fit(X_train_scaled, y_train)
# scoring on the scaled test set
print("Scaled test set accuracy: {:.2f}".format(svm.score(X_test_scaled, y_test))) # Scaled test set accuracy: 0.97
