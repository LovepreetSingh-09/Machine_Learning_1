# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 08:18:31 2019

@author: user
"""
# PCA (Principlal Component Analysis)

import graphviz
import mglearn
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
from IPython.display import display
from sklearn.model_selection import train_test_split

# Principal component analysis is a method that rotates the dataset in a way such that the rotated features are statistically uncorrelated. 
# Then selects only a subset of the new features, according to how important they are for explaining the data. 
# The most common application is to visualize the high-dimensional dataset

mglearn.plots.plot_pca_illustration()
plt.show()
#  1.) The first plot (top left) shows the original data points, colored to distinguish among them. 
# The algorithm proceeds by first finding the direction of maximum variance, labeled “Component 1.” This is the direction (or vector) in the data that contains most of the information, or in other words, the direction along which the features are most correlated with each other. 
# Then, the algorithm finds the direction that contains the most information while being orthogonal (at a right angle) to the first direction. In two dimensions, there is only one possible orientation that is at a right angle, but in higher-dimensional spaces there would be (infinitely) many orthogonal directions. 
# Although the two components are drawn as arrows, it doesn’t really matter where the head and the tail are; we could have drawn the first component from the center up to the top left instead of down to the bottom right. 
# The directions found using this process are called principal components, as they are the main directions of variance in the data. In general, there are as many principal components as original features.
#  2.) The second plot shows the same data, but now rotated so that the first principal component aligns with the x-axis and the second principal component aligns with the y-axis. Plot is First PC v/s Second PC
# Before the rotation, the mean was subtracted from the data, so that the transformed data is centered around zero. In the rotated representation found by PCA, the two axes are uncorrelated, meaning that the correlation matrix of the data in this representation is zero except for the diagonal.
#  3.) We can use PCA for dimensionality reduction by retaining only some of the principal components. We have kept only the first principal component and the second PC is dropped from the second plot. 
# This reduces the data from a two-dimensional dataset to a one-dimensional dataset. However, that instead of keeping only one of the original features, we found the most interesting direction ( top left to bottom right in the first panel) and kept this direction, the first principal component.
#  4.) Finally, we can undo the rotation and add the mean back to the data. The points are in the original feature space, but we kept only the information contained in the first principal component. 
# This transformation is sometimes used to remove noise effects from the data or visualize what part of the information is retained using the principal components.

# Simpler visualization by computing histograms of each of the features for the two classes, benign and malignant cancer :-
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
fig,axes=plt.subplots(15,2,figsize=(10,20))
malignant=cancer.data[cancer.target==0]
benign=cancer.data[cancer.target==1]
ax=axes.ravel()
for i in range(30):
    _,bins=np.histogram(cancer.data[:,i],bins=50)
    ax[i].hist(malignant[:,i],bins=bins,color=mglearn.cm3(0),alpha=0.5)
    ax[i].hist(benign[:,i],bins=bins,color=mglearn.cm3(2),alpha=0.5)
    ax[i].set_title(cancer.feature_names[i])
    ax[i].set_yticks(())
ax[0].set_xlabel("Feature magnitude") 
ax[0].set_ylabel("Frequency")
ax[0].legend(["malignant", "benign"], loc="best")
fig.tight_layout()
plt.show()
# Here we create a histogram for each of the features, counting how often a data point appears with a feature in a certain range (called a bin). 
# The histogram plot with least overlaps of 2 classe is the most informative and important feature
# While the plot with maximum overlap is the least informative and important feature

# Using PCA, we can capture the main interactions and get a slightly more complete picture. We can find the first two principal components, and visualize the data in this new two-dimensional space with a single scatter plot.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
scaler.fit(cancer.data) 
X_scaled = scaler.transform(cancer.data)
# We instantiate the PCA object, find the principal components by calling the fit method, and then apply the rotation and dimensionality reduction by calling transform. 
# By default, PCA only rotates (and shifts) the data, but keeps all principal components. To reduce the dimensionality of the data, we need to specify how many components we want to keep when creating the PCA object
from sklearn.decomposition import PCA
# keep the first two principal components of the data
pca=PCA(n_components=2)
# fit PCA model to breast cancer data 
pca.fit(X_scaled)
# transform data onto the first two principal components
X_pca=pca.transform(X_scaled)
print("Original shape: {}".format(str(X_scaled.shape))) 
print("Reduced shape: {}".format(str(X_pca.shape)))
# plot first vs. second principal component, colored by class
plt.figure(figsize=(8, 8))
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
plt.legend(cancer.target_names, loc="best")
plt.gca().set_aspect("equal") 
plt.xlabel("First principal component") 
plt.ylabel("Second principal component")
plt.show()

print("PCA component shape: {}".format(pca.components_.shape)) # PCA component shape: (2, 30)
# Each row in components_ corresponds to one principal component, and they are sorted by their importance (the first principal component comes first, etc.).
# The columns correspond to the original features attribute of the PCA
print("PCA components:\n{}".format(pca.components_)) 

plt.matshow(pca.components_, cmap='viridis')
plt.yticks([0, 1], ["First component", "Second component"])
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)),cancer.feature_names, rotation=60, ha='left')
plt.xlabel("Feature") 
plt.ylabel("Principal components")

