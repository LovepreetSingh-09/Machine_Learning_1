# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:12:00 2019

@author: user
"""

# Manifold Learning with t-SNE

# It is a class of algorithms for visualization that allow for much more complex mappings and often provide better visualizations. A particularly useful one is the t-SNE algorithm.
# Manifold learning algorithms are rarely used to generate more than two new features.
# Some of them, including t-SNE cannot be applied to a test set, they can only transform the data they were trained for. Manifold learning can be useful for exploratory data analysis, but is rarely used if the final goal is supervised learning.

# The idea behind t-SNE is to find a two-dimensional representation of the data that preserves the distances between points as best as possible.
# t-SNE starts with a random 2D representation for each data point and then tries to make points that are close in the original feature space closer and points that are far apart in the original feature space farther apart.
# t-SNE puts more emphasis on points that are close by rather than preserving distances between far-apart points.
# In other words, it tries to preserve the information indicating which points are neighbors to each other. 

import graphviz
import mglearn
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
from IPython.display import display
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_digits
digits=load_digits()
print(digits.keys())
print(digits.images.shape) # (1797, 8, 8)  1797 images of size 8 X 8
fig,axes=plt.subplots(2,5,figsize=(15,8),subplot_kw={'xticks':(),'yticks':()})
for ax,image in zip(axes.ravel(),digits.images):
    ax.imshow(image)
plt.show()

# 2D visualization of PCA
# Using PCA all the classes or digits are overlaping on each other which makes it impossible to distinguish between the classes
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(digits.data)
digits_pca=pca.transform(digits.data)
colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525","#A83683", "#4E655E", "#853541", "#3A3120", "#535D8E"]
plt.figure(figsize=(10,8))
plt.xlim(digits_pca[:,0].min(),digits_pca[:,0].max())
plt.ylim(digits_pca[:,1].min(),digits_pca[:,1].max())
print(len(digits.data)) # 1797
for i in range(len(digits.data)):
    plt.text(digits_pca[i,0],digits_pca[i,1],str(digits.target[i]),color=colors[digits.target[i]],fontdict={'weight': 'bold', 'size': 9})
plt.xlabel("First principal component") 
plt.ylabel("Second principal component")
plt.show()

# Using t-Sne
from sklearn.manifold import TSNE
tsne=TSNE(random_state=42)
digits_tsne=tsne.fit_transform(digits.data)
plt.figure(figsize=(15,12))
plt.xlim(digits_tsne[:,0].min(),digits_tsne[:,0].max()+1)
plt.ylim(digits_tsne[:,1].min(),digits_tsne[:,1].max()+1)
colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525","#A83683", "#4E655E", "#853541", "#3A3120", "#535D8E"]
for i in range(len(digits.data)):
    plt.text(digits_tsne[i,0],digits_tsne[i,1],str(digits.target[i]),color=colors[digits.target[i]],fontdict={'weight': 'bold', 'size': 9})
plt.xlabel("t-SNE feature 0") 
plt.xlabel("t-SNE feature 1")
plt.show()
# The result of t-SNE is quite remarkable. All the classes are quite clearly separated.
# The 1 and 9 are somewhat split up, but most of the classes form a single dense group.
# Keep in mind that this method has no knowledge of the class labels, it is completely unsupervised.
# Still, it can find a representation of the data in two dimensions that clearly separates the classes, based solely on how close points are in the original space.

