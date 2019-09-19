# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 23:46:51 2019

@author: user
"""

# DBSCAN (Density Based Spatial Clustering Algorithm with Noise)
# It doesn't need to set the no. of clusters by the user and can searate complex shapes and points that are not the part of any cluster.
# Slower than alggomerative and k-means but still works well on relatively large datastes.

import graphviz
import mglearn
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
from IPython.display import display
from sklearn.model_selection import train_test_split

# Points that are within a dense region are called core samples ( or core points)
# There are two parameters in DBSCAN: min_samples and eps.
# If there are at least min_samples many data points within a distance of eps to a given data point, that data point is classified as a core sample.
# Core samples that are closer to each other within the distance eps are put into the same cluster by DBSCAN.
# If there are less than min_samples points within distance eps of the starting point, this point is labeled as noise, meaning that it doesn’t belong to any cluster. 
# Boundary Boints -Points that are within distance eps of core points but not assigned the cluster label because they are not the core samples.
# The cluster grows until there are no more core samples within distance eps of the cluster.
# Then another point that hasn’t yet been visited is picked, and the same procedure is repeated.

# Like agglomerative clustering, DBSCAN does not allow predictions on new test data, so we will use the fit_predict method to perform clustering :-
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
X,y=make_blobs(random_state=0, n_samples=12)
dbscan = DBSCAN()
clusters=dbscan.fit_predict(X)
print("Cluster memberships:\n{}".format(clusters)) 
# As there is -1 cluster prediction for each data point which stands for noise.
# This is a consequence of the default parameter settings for eps and min_samples, which are not tuned for small toy datasets.
mglearn.plots.plot_dbscan()
plt.show()
# Noise points are shown in white while Core samples are shown as large markers and boundary points are displayed as smaller markers. 
# Increasing eps (going from left to right) means that more points will be included in a cluster. This makes clusters grow, but might also lead to multiple clusters joining into one.
# The parameter eps is somewhat more important, as it determines what it means for points to be “close.” Setting eps to be very small will mean that no points are core samples, and may lead to all points being labeled as noise. Setting eps to be very large will result in all points forming a single cluster.
# Setting eps implicitly controls how many clusters will be found. 
# Increasing min_samples ( going from top to bottom in the figure) means that fewer points will be core points, and more points will be labeled as noise.
# The min_samples setting mostly determines whether points in less dense regions will be labeled as outliers or as their own clusters. If you increase min_samples, anything that would have been a cluster with less than min_samples many samples will now be labeled as noise.
# Hence, min_samples therefore determines the minimum cluster size. 

# Finding a good setting for eps is sometimes easier after scaling the data using StandardScaler or MinMaxScaler, as using these scaling techniques will ensure that all features have similar ranges. 
# Running DBSCAN on the two_moons dataset perfectly separate 2 moons. 
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
# rescale the data to zero mean and unit variance
scaler = StandardScaler()
scaler.fit(X) 
X_scaled = scaler.transform(X)
dbscan = DBSCAN()
clusters = dbscan.fit_predict(X_scaled)
# plot the cluster assignments
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm2, s=60)
plt.xlabel("Feature 0") 
plt.ylabel("Feature 1")
plt.show()
# Default eps = 0.5 which perfectly makes 2 clusters.
# Default min_samples = 5
# Decreasing eps will make more clusters like eps = 0.2 will make 8 clusters.
# Increasing eps will make the whole dataset as 1 cluster.

