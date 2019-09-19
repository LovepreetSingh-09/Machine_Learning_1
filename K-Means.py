# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:22:44 2019

@author: user
"""

# Clustering :-
# Clustering is the task of partitioning the dataset into groups, called clusters.
# The goal is to split up the data in such a way that points within a single cluster are very similar and points in different clusters are different.
# Similarly to classification algorithms, clustering algorithms assign (or predict) a number to each data point, indicating which cluster a particular point belongs to. 

# K-Means Clustering :-
# kmeans clustering is one of the simplest and most commonly used clustering algorithms.
# It tries to find cluster centers that are representative of certain regions of the data.
# The algorithm alternates between two steps: 
#   1.) assigning each data point to the closest cluster center
#   2.) Then setting each cluster center as the mean of the data points that are assigned to it.
# The algorithm is finished when the assignment of instances to clusters no longer changes. 

import graphviz
import mglearn
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
from IPython.display import display
from sklearn.model_selection import train_test_split

# kmeans on synthetic dataset:-
mglearn.plots.plot_kmeans_algorithm()
plt.show()
# Boundaries of kmeans :-
mglearn.plots.plot_kmeans_boundaries()
plt.show()

# # kmeans on synthetic dataset:-
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
# generate synthetic two-dimensional data
X,y=make_blobs(random_state=42)
kmeans=KMeans(n_clusters=3)
# During the algorithm, each training data point in X is assigned a cluster label.
kmeans.fit(X)
# You can find these labels in the kmeans.labels_ attribute
print("Cluster memberships:\n{}".format(kmeans.labels_))
# Running predict is same as the labels_
print('Predictions : ',kmeans.predict(X))

# In the example of clustering, face images that we discussed before, it might be that the cluster 3 found by the algorithm contains only faces of your friend Bela.
# You can only know that after you look at the pictures, though, and the number 3 is arbitrary.
# The only information the algorithm gives you is that all faces labeled as 3 are similar.

# The cluster centers are stored in the cluster_centers_ attribute, and we plot them as triangles
# Running the algorithm again might result in a different numbering of clusters because of the random nature of the initialization.
mglearn.discrete_scatter(X[:,0],X[:,1],kmeans.labels_,markers='o')
# Here we didn't use kmeans.labels_ with centers because centers are only 3 but labels are as many as data points in the data
mglearn.discrete_scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],[0,1,2],markers='^',markeredgewidth=6)
plt.show()

fig,axes=plt.subplots(1,2,figsize=(15,7))
# No. of Cluster Centers:-
# Cluster_centers_=2
kmeans=KMeans(n_clusters=2)
kmeans.fit(X)
mglearn.discrete_scatter(X[:,0],X[:,1],kmeans.labels_,ax=axes[0])
# Cluster_centers_=5
kmeans=KMeans(n_clusters=5)
kmeans.fit(X)
mglearn.discrete_scatter(X[:,0],X[:,1],kmeans.labels_,ax=axes[1])
plt.show()

# Failure Cases of k-means :-
#     Even if you know the “right” number of clusters for a given dataset, k-means might not always be able to recover them.
#     Each cluster is defined solely by its center, which means kmeans can only capture relatively simple shapes.
# 1.) k-means also assumes that all clusters have the same “diameter” in some sense; it always draws the boundary between clusters to be exactly in the middle between the cluster centers.
# That can sometimes lead to surprising and unwanted results
X_varied,y_varied=make_blobs(n_samples=200,cluster_std=[1,2.5,0.5],random_state=170)
# cluster_std is the standard deviation of clusters
y_pred=KMeans(n_clusters=3).fit_predict(X_varied)
mglearn.discrete_scatter(X_varied[:,0],X_varied[:,1],y_pred)
plt.legend(["cluster 0", "cluster 1", "cluster 2"], loc='best')
plt.xlabel("Feature 0") 
plt.ylabel("Feature 1")
plt.show()
# Here even the 1st and 3rd clusters are very less spread but cluster 2 is very much spread but there are some points of 1st and 3rd clusters in the 2nd cluster region
# This is because of the same diameter

# 2.) k-means also assumes that all directions are equally important for each cluster.
# Here a two-dimensional dataset where there are three clearly separated parts in the data which are stretched toward the diagonal.
# As k-means only considers the distance to the nearest cluster center, it can’t handle this kind of data.
X,y=make_blobs(n_samples=600,random_state=170)
rng=np.random.RandomState(74) # Random state fixed the no. between 0 and 1
print(rng) # an object for making array of fixed no. between 0 and 1
# transform the data to be stretched 
transformation=rng.normal(size=(2,2))
print(transformation) # (2,2) shape array of no. between 0 and 1
X=np.dot(X,transformation)
kmeans=KMeans(n_clusters=3)
kmeans.fit(X)
y_pred=kmeans.predict(X)
print(kmeans.cluster_centers_[:,0],'\n',kmeans.cluster_centers_[:,1])
mglearn.discrete_scatter(X[:,0],X[:,1],kmeans.labels_,markers='o')
mglearn.discrete_scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],[0,1,2],markers='^',markeredgewidth=5)
plt.xlabel("Feature 0") 
plt.ylabel("Feature 1")
plt.show()

# 3.) k-means also performs poorly if the clusters have more complex shapes, like the two_moons data we encountered earlier
from sklearn.datasets import make_moons
# noise = standard deviation of gaussian noise added to the data
# random_state = determines random no. generation for dataset shuffling and noise. 
# It actually does shuffling, more the value more will be the shuffling
X,y=make_moons(n_samples=200,noise=0.05,random_state=0)
kmeans=KMeans(n_clusters=2)
kmeans.fit(X)
y_pred=kmeans.predict(X)
plt.scatter(X[:,0],X[:,1],c=y_pred,cmap=mglearn.cm2,s=60,marker='o')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c=[mglearn.cm2(0),mglearn.cm2(1)],s=100,marker='^',linewidth=2,edgecolor='g')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

# There are only just 2 custer centers 
# when kmeans.predict(X) will pass through it, an array (200,2) wil be made containing 2 types of values of centers.
# Those value represent the center of the cluster which they represent which is 0 or 1 
print(kmeans.cluster_centers_[kmeans.predict(X)].shape)
print(kmeans.cluster_centers_.shape) # (2,2) 1st 2 = no. of clusters; 2nd 2 = dimenions of the cluster 
print(kmeans.predict(X))
