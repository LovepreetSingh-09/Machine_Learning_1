# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:03:57 2019

@author: user
"""
# K-Means as decompositon method for face reconstruction
# Vector Quantization :-
# In  K-means, each point being represented using only a single component, which is given by the cluster center.
# This view of k-means as a decomposition method, where each point is represented using a single component, is called vector quantization.

import graphviz
import mglearn
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
from IPython.display import display
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from sklearn.datasets import fetch_lfw_people
people=fetch_lfw_people(min_faces_per_person=20,resize=0.7)
image_shape=people.images[0].shape
mask=np.zeros(people.target.shape,dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target==target)[0][:50]]=1
X_people=people.data[mask]
y_people=people.target[mask]
X_people = X_people / 255.0
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
nmf = NMF(n_components=100, random_state=0)
nmf.fit(X_train)
pca = PCA(n_components=100, random_state=0)
pca.fit(X_train)
kmeans = KMeans(n_clusters=100, random_state=0) 
kmeans.fit(X_train)
rcon_pca=pca.inverse_transform(pca.transform(X_test))
rcon_nmf=np.dot(nmf.transform(X_test),nmf.components_)
# There are 100 cluster centers represent the each data point of X_test
# So 516 data points have its own cluster center.
# Cluster center is represented by the same no. of features or dimensions as training data has.
rcon_kmeans=kmeans.cluster_centers_[kmeans.predict(X_test)]
print(np.unique(kmeans.predict(X_test))) # shape(100,) from 0 to 99 .
print(rcon_kmeans.shape) # (516, 5655)
print(kmeans.cluster_centers_.shape) # (100, 5655)
fig, axes = plt.subplots(3, 5, figsize=(8, 8) ,subplot_kw={'xticks': (), 'yticks': ()})
fig.suptitle("Extracted Components") 
# Components reshaping
# iterating in axes works row-wise but axes.T works column wise because of transpose
for ax, comp_kmeans, comp_pca, comp_nmf in zip(axes.T, kmeans.cluster_centers_, pca.components_, nmf.components_):
     ax[0].imshow(comp_kmeans.reshape(image_shape))     
     ax[1].imshow(comp_pca.reshape(image_shape), cmap='viridis')     
     ax[2].imshow(comp_nmf.reshape(image_shape))
axes[0, 0].set_ylabel("kmeans") 
axes[1, 0].set_ylabel("pca") 
axes[2, 0].set_ylabel("nmf")
plt.show()
fig, axes = plt.subplots(4, 5, subplot_kw={'xticks': (), 'yticks': ()} ,figsize=(8, 8)) 
fig.suptitle("Reconstructions")
# Reconstruction
for ax,x_test,rp,rn,rk in zip(axes.T,X_test,rcon_pca,rcon_nmf,rcon_kmeans):
    ax[0].imshow(x_test.reshape(image_shape))
    ax[1].imshow(rp.reshape(image_shape))
    ax[2].imshow(rn.reshape(image_shape))
    ax[3].imshow(rk.reshape(image_shape))
axes[0, 0].set_ylabel("original") 
axes[1, 0].set_ylabel("kmeans") 
axes[2, 0].set_ylabel("pca") 
axes[3, 0].set_ylabel("nmf")    
plt.show()
# As we can see that kmeans reconstruct faces better than pca and nmf of the data
# An interesting aspect of vector quantization using k-means is that we can use many more clusters than input dimensions to encode our data but data points or samples should be more than the no. of clusters. 

# make_moons using 10 clusters
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(X) 
y_pred = kmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=60, cmap='Paired')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=60, marker='^', c=range(kmeans.n_clusters), linewidth=2, cmap='Paired')
plt.xlabel("Feature 0") 
plt.ylabel("Feature 1")
print("Cluster memberships:\n{}".format(y_pred))

# Increasing features or dimensions 
distance_features = kmeans.transform(X)
print("Distance feature shape: {}".format(distance_features.shape)) 
# distance to each cluster center of every data point.
# The feature having lowest value will be the cluster of that data point
print("Distance features:\n{}".format(distance_features)) 
# The output of both the following codes will be same
print("Distance features:\n{}".format(np.argmin(distance_features,axis=1)))
print(kmeans.predict(X))

# k-means is a very popular algorithm for clustering, not only because it is relatively easy to understand and implement, but also because it runs relatively quickly.
# kmeans scales easily to large datasets
# One of the drawbacks of k- means is that it relies on a random initialization, which means the outcome of the algorithm depends on a random seed.
# Further downsides of k- means are the relatively restrictive assumptions made on the shape of clusters, and the requirement to specify the number of clusters you are looking for (which might not be known in a real-world application).
