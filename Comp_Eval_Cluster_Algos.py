# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 00:31:41 2019

@author: user
"""

# Comp adnd Evaluating Clustering Algorithms

import graphviz
import mglearn
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
from IPython.display import display
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

# Comparing Models with Ground Truth :-
# Ground Truth is the reality you want your model to predict.No model gives 100 % accuracy but we want our model to be as close as possible to the Ground Truth.
# There are metrics that can be used to assess the outcome of a clustering algorithm relative to a ground truth clustering.
# The most important ones being the adjusted rand index (ARI) and normalized mutual information (NMI), which both provide a quantitative measure with an optimum of 1 and a value of 0 for unrelated clusterings(though the ARI can become negative).
# ARI = adjusted_rand_score
# NMI = normalized_mutual_info_score
# We will use ARI
from sklearn.metrics.cluster import adjusted_rand_score
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
# rescale the data to zero mean and unit variance
scaler = StandardScaler()
scaler.fit(X) 
X_scaled = scaler.transform(X)
fig, axes = plt.subplots(1, 4, figsize=(15, 3),subplot_kw={'xticks': (), 'yticks': ()})
# make a list of algorithms to use
algorithms = [KMeans(n_clusters=2),AgglomerativeClustering(n_clusters=2),DBSCAN()]
# create a random cluster assignment for reference 
random_state=np.random.RandomState(seed=0)
# Makes an array of len(200) of X representing random values 0 and 1
random_clusters=random_state.randint(low=0,high=2,size=len(X))
print(random_clusters)
# plot random assignment
axes[0].scatter(X_scaled[:,0],X_scaled[:,1],c=random_clusters,s=60,cmap=mglearn.cm3)
axes[0].set_title('Random Assignment : {:.2f}'.format(adjusted_rand_score(y,random_clusters)))
for ax,algo in zip(axes[1:],algorithms):
    ypred=algo.fit_predict(X_scaled)
    ax.scatter(X_scaled[:,0],X_scaled[:,1],c=ypred,cmap=mglearn.cm2,s=60)
    ax.set_title('{} : {:.2f}'.format(algo.__class__.__name__,adjusted_rand_score(y,ypred)))
plt.show()
# Random Assignment = 0.0 ; KMeans = 0.50 ; AgglomerativeClustering = 0.61 ; DBSCAN = 1.0

# The problem in using accuracy_score is that it requires the assigned cluster labels to exactly match the ground truth.
# However, the cluster labels themselves are meaningless—the only thing that matters is which points are in the same cluster.
# There is randomization so sometimes a cluster will be labeled as 0 and sometimes 1.
# So. ARI gives score on the basis of points in the same cluster irrespective of labels
from sklearn.metrics import accuracy_score
# these two labelings of points correspond to the same clustering
clusters1 = [0, 0, 1, 1, 0] 
clusters2 = [1, 1, 0, 0, 1]
# accuracy is zero, as none of the labels are the same
print("Accuracy: {:.2f}".format(accuracy_score(clusters1, clusters2)))
# adjusted rand score is 1, as the clustering is exactly the same 
print("ARI: {:.2f}".format(adjusted_rand_score(clusters1, clusters2)))

# Without Ground Truth :
# Although we have just shown one way to evaluate clustering algorithms but in practice, there is a big problem with using measures like ARI
# When applying clustering algorithms, there is usually no ground truth to which to compare the results.
# If we knew the right clustering of the data, we could use this information to build a supervised model like a classifier.
# Therefore, using metrics like ARI and NMI usually only helps in developing algorithms, not in assessing success in an application.
# There are scoring metrics for clustering that don’t require ground truth, like the silhouette coefficient.
# However, these often don’t work well in practice. The silhouette score computes the compactness of a cluster, where higher is better, with a perfect score of 1.
# While compact clusters are good, compactness doesn’t allow for complex shapes.
# Previous Comparing of Algorithms using silhouette score
from sklearn.metrics.cluster import silhouette_score
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
# rescale the data to zero mean and unit variance
scaler = StandardScaler()
scaler.fit(X) 
X_scaled = scaler.transform(X)
fig, axes = plt.subplots(1, 4, figsize=(15, 3),subplot_kw={'xticks': (), 'yticks': ()})
# make a list of algorithms to use
algorithms = [KMeans(n_clusters=2),AgglomerativeClustering(n_clusters=2),DBSCAN()]
# create a random cluster assignment for reference 
random_state=np.random.RandomState(seed=0)
# Makes an array of len(200) of X representing random values 0 and 1
random_clusters=random_state.randint(low=0,high=2,size=len(X))
print(random_clusters)
# plot random assignment
axes[0].scatter(X_scaled[:,0],X_scaled[:,1],c=random_clusters,s=60,cmap=mglearn.cm3)
axes[0].set_title('Random Assignment : {:.2f}'.format(adjusted_rand_score(y,random_clusters)))
for ax,algo in zip(axes[1:],algorithms):
    ypred=algo.fit_predict(X_scaled)
    ax.scatter(X_scaled[:,0],X_scaled[:,1],c=ypred,cmap=mglearn.cm2,s=60)
    ax.set_title('{} : {:.2f}'.format(algo.__class__.__name__,silhouette_score(X_scaled,ypred)))
plt.show()
# Random Assignment = 0.0 ; KMeans = 0.49 ; AgglomerativeClustering = 0.46 ; DBSCAN = 0.38
# This is because of the compactness
# Even if we get a very robust clustering, or a very high silhouette score, we still don’t know if there is any semantic meaning in the clustering, or whether the clustering reflects an aspect of the data that we are interested in.

# Comparing Algorithms on Face Dataset :-
# We hope to find groups of similar faces—say, men and women, or old people and young people, or people with beards and without.
# Let’s say we cluster the data into two clusters, and all algorithms agree about which points should be clustered together.
# We still don’t know if the clusters that are found correspond in any way to the concepts we are interested in.
# It could be that they found side views versus front views or pictures taken at night versus pictures taken during the day or pictures taken with iPhones versus pictures taken with Android phones.
# The only way to know whether the clustering corresponds to anything we are interested in is to analyze the clusters manually.

# 1.) Analyzing the faces dataset with DBSCAN :-
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
from sklearn.decomposition import PCA
# We will use the eigenface representation of the data, as produced by PCA(whiten=True), with 100 components.
pca = PCA(n_components=100, whiten=True, random_state=0)
pca.fit(X_people)
X_pca = pca.transform(X_people)
dbscan=DBSCAN()
labels=dbscan.fit_predict(X_pca)
print("Unique labels: {}".format(np.unique(labels))) # Unique labels: [-1]
# Here, all the points are noise. So, now we will make either eps higher or min_samples lower
dbscan=DBSCAN(min_samples=3)
labels=dbscan.fit_predict(X_pca)
print("Unique labels: {}".format(np.unique(labels))) # Unique labels: [-1]
# Still after reducing min_samples all the data points are noise
# Now increase the eps
dbscan=DBSCAN(min_samples=3,eps=15)
labels=dbscan.fit_predict(X_pca)
print("Unique labels: {}".format(np.unique(labels))) # Unique labels: [-1  0]
print(np.unique(labels+1)) # [0 1 ]
# Count number of points in all clusters and noise.
# bincount doesn't allow negative numbers, so we need to add 1.
# The first number in the result corresponds to noise points. 
print('No. of Points per Cluster : {}'.format(np.bincount(labels+1))) #  [ 32 2031]
# So there are just 32 noise points and rwmaining are in a cluster
# Images which act as noise:-
print(labels==-1) #  Gives an 1D array in bolean True / False
noise=X_people[labels==-1] # Noise gives those data points where labels==-1 is True
print(noise.shape) # (32, 5655)
fig, axes = plt.subplots(3, 9, subplot_kw={'xticks': (), 'yticks': ()} ,figsize=(12, 4)) 
for ax,image in zip(axes.ravel(),noise):
    ax.imshow(image.reshape(image_shape),vmin=0,vmax=1) # vmin and vmax here doesn't make any difference
    # vmin amd vmax defines the data range that the colormap covers
plt.show()
# As we can see from the images why these are labeled noise because these images are odd one out.
# In these images some has drinking wine in front of their face or hat front of face or other things happen by which the faces are not recognized properly.
# This kind of analysis—trying to find “the odd one out”—is called outlier detection. 
# If this was a real application, we might try to do a better job of cropping images, to get more homogeneous data.
# There is little we can do about people in photos sometimes wearing hats, drinking, or holding something in front of their faces, but it’s good to know that these are issues in the data that any algorithm we might apply needs to handle.
# If we want to find more interesting clusters than just one large one, we need to set eps smaller, somewhere between 15 and 0.5 (the default).
for eps in [1,3,5,7,9,11,13]:
    print('eps : {}'.format(eps))
    dbscan=DBSCAN(min_samples=3,eps=eps)
    cluster=dbscan.fit_predict(X_pca)
    print('No. of Unique Clusters : {}'.format(len(np.unique(cluster))))
    print('Size of Clusters : {}\n'.format(np.bincount(cluster+1)))
# Till eps=5(low settings) all images are noise.
# For eps=7 we got 13 small clusters and many noise points
# For eps=9 we still get many noise points, but we get one big cluster and some smaller clusters.
# From eps=11, we get only one large cluster and noise.
# What is interesting to note is that there is never more than one large cluster. 
# This indicates that there are not two or three different kinds of face images in the data that are very distinct, but rather that all images are more or less equally similar to (or dissimilar from) the rest.   
# The results for eps=7 look most interesting, with many small clusters. So, let's take a look at its images :-
dbscan = DBSCAN(min_samples=3, eps=7) 
labels = dbscan.fit_predict(X_pca)
# labels= -1 to 12
print(max(labels)) # 12
print(max(labels)+1)  # 13
for cluster in range(max(labels)+1):
    mask=labels==cluster
    # Makes 1D array in True/False where cluster matches in the labels 
    print(mask)
    n_images=np.sum(mask) # No. of images of the particular cluster
    print(n_images)
    fig, axes = plt.subplots(1, n_images, figsize=(n_images * 1.5, 4),subplot_kw={'xticks': (), 'yticks': ()}) 
    for image,target,ax in zip(X_people[mask],y_people[mask],axes.ravel()):
        ax.imshow(image.reshape(image_shape),vmin=0,vmax=1)
        ax.set_title(people.target_names[target].split()[-1])
plt.show()
# Some of the clusters correspond to people with very distinct faces (within this dataset), such as Sharon or Koizumi.
# Within each cluster, the orientation of the face is also quite fixed, as well as the facial expression.
# Some of the clusters contain faces of multiple people, but they share a similar orientation and expression.


# 2.) Analyzing the faces dataset with k-means :-
# We saw that it was not possible to create more than one big cluster using DBSCAN.
# Agglomerative clustering and k-means are much more likely to create clusters of even size, but we do need to set a target number of clusters.
# We could set the number of clusters to the known number of people in the dataset, though it is very unlikely that an unsupervised clustering algorithm will recover them.
# Instead, we can start with a low number of clusters, like 10, which might allow us to analyze each of the clusters
km = KMeans(n_clusters=10, random_state=0)
labels_km = km.fit_predict(X_pca)
print("Cluster sizes k-means: {}".format(np.bincount(labels_km)))
#  [155 175 238  75 358 257  91 219 323 172]
# In k-means and agglomerative there is no noise so every data point refers to a cluster
# We can further analyze the outcome of k-means by visualizing the cluster centers.
# As we clustered in the representation produced by PCA, we need to rotate the cluster centers back into the original space to visualize them, using pca.inverse_transform.
# There are 10 cluster centers of X_pca of dimension 100
# By inverse_transform the new values of cluster center will be in dimension 5655
fig, axes = plt.subplots(2, 5, subplot_kw={'xticks': (), 'yticks': ()} ,figsize=(12, 4)) 
for ax,center in zip(axes.ravel(),km.cluster_centers_):
    ax.imshow(pca.inverse_transform(center).reshape(image_shape),vmin=0,vmax=1)
plt.show()
# The cluster centers found by k-means are very smooth versions of faces as each center is an average of no. of face images.
# Working with a reduced PCA representation adds to the smoothness of the images.
# The clustering seems to pick up on different orientations of the face, different expressions (the third cluster center seems to show a smiling face), and the presence of shirt collars (see the second-to-last cluster center).
# For a more detailed view, we have shown each cluster center the five most typical images in the cluster(that are closest to the cluster center) and the five most atypical images in the cluster(that are furthest from the cluster center)
mglearn.plots.plot_kmeans_faces(km,pca,X_pca,X_people,y_people,people.target_names)
plt.show()
# Here, there is smiling faces for the third cluster and also the importance of orientation for the other clusters.
# The “atypical” points are not very similar to the cluster centers and their assignment seems somewhat arbitrary. 


# 3.) Analyzing the faces dataset with Agglomerative clustering :-
# extract clusters with ward agglomerative clustering 
from scipy.cluster.hierarchy import dendrogram,ward
agglomerative = AgglomerativeClustering(n_clusters=10) 
labels_agg = agglomerative.fit_predict(X_pca) 
print("Cluster sizes agglomerative clustering: {}".format(np.bincount(labels_agg))) 
# [169 660 144 329 217  85  18 261  31 149]
# ARI score of labels of kmeans and labels of agglomerative
print("ARI: {:.2f}".format(adjusted_rand_score(labels_agg, labels_km))) # 0.09
# An ARI of only 0.13 means that the two clusterings labels_agg and labels_km have little in common. 
linkage_array=ward(X_pca)
print(linkage_array.shape) # (2062, 4)
# 1st and 2nd column shows the index clusters to merge
# 3rd column shows the distance between those merged clusters
# 4th column shows the no. of cluster points within that merged cluster
# now we plot the dendrogram for the linkage_array
# containing the distances between clusters
plt.figure(figsize=(20, 5))
dendrogram(linkage_array,p=7,truncate_mode='level', no_labels=True)
# p denotes the leaf nodes in the plot
plt.xlabel("Sample index") 
plt.ylabel("Cluster distance")
plt.show()
# Here doesn’t appear to be a particular number of clusters that is a good fit.
# This is not surprising, given the results of DBSCAN, which tried to cluster all points together.
# Do similar to recover faces as we did for DBSCAN
n_clusters = 10
for cluster in range(n_clusters):
     mask = labels_agg == cluster
     fig, axes = plt.subplots(1, 10, subplot_kw={'xticks': (), 'yticks': ()},figsize=(15, 8))     
     axes[0].set_ylabel(np.sum(mask))
     for image,label,ax in zip(X_people[mask],y_people[mask],axes):         
         ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)         
         ax.set_title(people.target_names[label].split()[-1],fontdict={'fontsize': 9})
plt.show()
# While some of the clusters seem to have a semantic theme, many of them are too large to be actually homogeneous.
# To get more homogeneous clusters, we can run the algorithm again, this time with 40 clusters and pick out some of the clusters that are particularly interesting.
# extract clusters with ward agglomerative clustering 
agglomerative = AgglomerativeClustering(n_clusters=40) 
labels_agg = agglomerative.fit_predict(X_pca)
print("cluster sizes agglomerative clustering: {}".format(np.bincount(labels_agg)))
n_clusters = 40
for cluster in [10, 13, 19, 22, 36]: # hand-picked "interesting" clusters
    mask = labels_agg == cluster
    fig, axes = plt.subplots(1, 15, subplot_kw={'xticks': (), 'yticks': ()} ,figsize=(15, 8))
    cluster_size = np.sum(mask)
    axes[0].set_ylabel("#{}: {}".format(cluster, cluster_size))     
    for image, label, ax in zip(X_people[mask],y_people[mask],axes):         
            ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)        
            ax.set_title(people.target_names[label].split()[-1] ,fontdict={'fontsize': 9})     
plt.show()
# Here, the clustering seems to have picked up on “dark skinned and smiling,” “collared shirt,” “smiling woman,” “Hussein,” and “high forehead.”
for i in range(40,15,-1):
    print(i,'\n')

# All algorithms in scikit-learn, whether preprocessing, supervised learning, or unsupervised learning algorithms, are implemented as classes.
# These classes are called estimators in scikitlearn.





