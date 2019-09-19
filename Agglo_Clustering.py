# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:18:48 2019

@author: user
"""

# Agglomerative Clustering 
# Agglomerative clustering refers to a collection of clustering algorithms that all build upon the same principles: 
# The algorithm starts by declaring each point its own cluster, and then merges the two most similar clusters until some stopping criterion is satisfied.
# The stopping criterion implemented in scikit-learn is the number of clusters, so similar clusters are merged until only the specified number of clusters are left. 

import graphviz
import mglearn
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
from IPython.display import display
from sklearn.model_selection import train_test_split

mglearn.plots.plot_agglomerative_algorithm()
plt.show()
# There are several linkage criteria that specify how exactly the “most similar cluster” is measured. This measure is always defined between two existing clusters. The following three choices are implemented in scikit-learn:
# 1.) Ward :-    The default choice, ward picks the two clusters to merge such that the variance within all clusters increases the least. This often leads to clusters that are relatively equally sized.
# 2.) Average:-  Average linkage merges the two clusters that have the smallest average distance between all their points.
# 3.) Complete:- Complete linkage (also known as maximum linkage) merges the two clusters that have the smallest maximum distance between their points.
# Ward works on most datasets, and we will use it in our examples. If the clusters have very dissimilar numbers of members (if one is much bigger than all the others, for example), average or complete might work better.

# It cannot make predictions for the new data points so, it has no predict method.
# But we can use fit_predict method.
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
X, y = make_blobs(random_state=1)
agg = AgglomerativeClustering(n_clusters=3) 
assign=agg.fit_predict(X)
print(assign)
mglearn.discrete_scatter(X[:,0],X[:,1],assign)
plt.legend(["Cluster 0", "Cluster 1", "Cluster 2"], loc="best")
plt.xlabel("Feature 0") 
plt.ylabel("Feature 1")
plt.show()
# As expected, the algorithm recovers the clustering perfectly but it requires to specify the no. of clusters

# Hierarchical Clustering :- 
# Agglomerative clustering produces hierarchical clustering where every point makes a journey from being a single point cluster to belonging to some final cluster.
# Each intermediate step provides a clustering of the data (with a different number of clusters).
# It is sometimes helpful to look at all possible clusterings jointly.
# The next example shows an overlay of all the possible clusterings :-
mglearn.plots.plot_agglomerative()
plt.show()
# But this Hierarchical Clustering works on only 2D data.
# For multi-dimensional data , Dendrogram is used for that data's hierarchical clustering.
# Scikit doesn't have dendogram but Scipy has. There is some different interface of implementation of algorithm in Scipy.
from scipy.cluster.hierarchy import dendrogram,ward
X, y = make_blobs(random_state=0, n_samples=12)
# Apply the ward clustering to the data array X
# The SciPy ward function returns an array that specifies the distances
linkage_array=ward(X)
print(linkage_array)
# 1st and 2nd column shows the index clusters to merge
# 3rd column shows the distance between those merged clusters
# 4th column shows the no. of cluster points within that merged cluster
print(linkage_array.shape) # (11, 4)
# Now Dendrogram makes the plot
dendrogram(linkage_array)
# Mark the cuts in the tree that signify two or three clusters
ax=plt.gca()
bounds=ax.get_xbound()
print(bounds) # (0.0, 120.0)
ax.plot(bounds,[7.25,7.25],'--',c='k')
ax.plot(bounds,[4,4],'--',c='k')
ax.text(bounds[1],7.25,'Two Clusters',va='center',fontdict={'size':15})
ax.text(bounds[1],4,'Three Clusters',va='center',fontdict={'size':15})
plt.xlabel("Sample index") 
plt.ylabel("Cluster distance")
plt.show()
# The dendrogram shows data points as points on x-axis (numbered from 0 to 11).
# Then, a tree is plotted with these points (representing single-point clusters) as the leaves, and a new node parent is added for each two clusters that are joined.
# The y-axis in the dendrogram doesn’t just specify when in the agglomerative algorithm two clusters get merged but also the length of each branch that how far apart the merged clusters are. 
# The branches are the vertical lines and they rpresent distance between 2 clusters.

# Agglomeartive Clustering still fails to separate complex shapes like 2 moons
