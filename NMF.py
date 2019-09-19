# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 11:08:03 2019

@author: user
"""

# NMF ( Non-Negative Matrix Factorization )
# An Unsupervised algorithm used for feature extraction and dimensionality reduction
# NMF is mainly used for finding interesting patterns within the data.
# As in PCA, we are trying to write each data point as a weighted sum of some components.
# But whereas in PCA we wanted components that were orthogonal and that explained as much variance of the data as possible.
# But in NMF, we want the components and the coefficients to be nonnegative means greater than or equal to zero.
# Consequently, this method can only be applied to data where each feature is non-negative, as a non-negative sum of non-negative components cannot become negative.

# The basic idea behind nmf is the weighted sum of X_train_nmf and nmf.components_ to recover the original matrices X_train.
# The weighted sum or dot product will give approximate or very close value of X_train. 
# X_train_nmf or X_test_nmf act as weights or coefficient and nmf.components_ are components which are extracted from nmf.fit() method.
# AS this NMF, so weights and components should be non-negative

import graphviz
import mglearn
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# NMF can identify the original components that make up the combined data.
# Overall, NMF leads to more interpretable components than PCA, as negative components and coefficients can lead to hard-to-interpret cancellation effects.
# The eigenfaces contain both positive and negative parts and as we mentioned in the description of PCA, the sign is actually arbitrary.
# Before we apply NMF to the face dataset, let’s briefly revisit the synthetic data.

# In NMF, firstly we should have to ensure that the data is positive
# This means where the data lies relative to the origin (0, 0) actually matters for NMF.
# Therefore, you can think of the non-negative components that are extracted as directions from (0, 0) toward the data.
# Example of 2D toy data on NMF
mglearn.plots.plot_nmf_illustration()
plt.show()
# For NMF with two components it is clear that all points in the data can be written as a positive combination of the two components.
# If there are enough components to perfectly reconstruct the data (as many components as there are features), the algorithm will choose directions that point toward the extremes of the data.
# If we only use a single component, NMF creates a component that points toward the mean, as pointing there best explains the data. 
# Components in NMF are also not ordered in any specific way, all components play an equal part.
# NMF uses a random initialization, which might lead to different results depending on the random seed.
# In relatively simple cases with two components, where all the data can be explained perfectly, the randomness has little effect. In more complex situations, there might be more drastic changes.


from sklearn.datasets import fetch_lfw_people
people=fetch_lfw_people(min_faces_per_person=20,resize=0.7)
image_shape=people.images[0].shape
fig,axes=plt.subplots(2,5,figsize=(15,8),subplot_kw={'xticks':(),'yticks':()}) # subplot_kw is for disabling x and y ticks
for target,image,ax in zip(people.target,people.images,axes.ravel()):
    ax.imshow(image)
    ax.set_title(people.target_names[target])
    # we can also use the following for disabling x and y ticks
    ax.set_xticks(())
    ax.set_yticks(())
plt.show()
mask=np.zeros(people.target.shape,dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target==target)[0][:50]]=1
X_people=people.data[mask]
y_people=people.target[mask]
X_people = X_people / 255.0
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)

from sklearn.decomposition import NMF
nmf=NMF(n_components=15, random_state=0)
nmf.fit(X_train)
X_train_nmf = nmf.transform(X_train)
X_test_nmf = nmf.transform(X_test)

# Appling NMF to face images :-
# The main parameter of NMF is how many components we want to extract. 
# Usually this is lower than the number of input features (otherwise, the data could be explained by making each pixel a separate component).
# mglearn.plots.plot_nmf_faces(X_train,X_test,image_shape)
plt.show()
# The quality of the back-transformed data is similar to when using PCA, but slightly worse.
# This is expected, as PCA finds the optimum directions in terms of reconstruction. NMF is usually not used for its ability to reconstruct or encode data, but rather for finding interesting patterns within the data.
fig,axes=plt.subplots(3,5,figsize=(15,8),subplot_kw={'xticks':(),'yticks':()})
print(nmf.components_.shape) # (15, 5655)
for i,(ax,component) in enumerate(zip(axes.ravel(),nmf.components_)):
    ax.imshow(component.reshape(image_shape))
    ax.set_title('{} componets'.format(i))
plt.show()
# These components are all positive, and so resemble prototypes of faces much more so than the components shown for PCA.
# Here, component 3 shows a face rotated somewhat to the right, while component 7 shows a face somewhat rotated to the left.

# Now plot the images for which component 3 and component 7 are strong
comp=3
# Get the ascending sorted location or position of maximum value of component in an array and then reverses the array
inds=np.argsort(X_train_nmf[:,comp])[::-1]
print(inds.shape) # (1547,)
fig,axes=plt.subplots(3,5,figsize=(15,8),subplot_kw={'xticks':(),'yticks':()})
fig.suptitle('Component 3 images')
for i,(ind,ax) in enumerate(zip(inds,axes.ravel())):
    ax.imshow(X_train[ind].reshape(image_shape))
plt.show()
# For component 7
comp=7
# Get the ascending sorted location or position of maximum value of component in an array and then reverses the array
inds=np.argsort(X_train_nmf[:,comp])[::-1] # X_train_nmf contains the coefficient for the components
print(inds.shape) # (1547,)
fig,axes=plt.subplots(3,5,figsize=(15,8),subplot_kw={'xticks':(),'yticks':()})
fig.suptitle('Component 7 images')
for i,(ind,ax) in enumerate(zip(inds,axes.ravel())):
    ax.imshow(X_train[ind].reshape(image_shape))
plt.show()
# faces that have a high coefficient for component 3 are faces looking to the right.
# while faces with a high coefficient for component 7 are looking to the left .

# Let’s walk through one example on synthetic data.
# Synthetic data is information that is artificially manufactured rather than real world events. 
# Here, we are interested in a signal that is a combination of three different sources 
S=mglearn.datasets.make_signals()
print(S.shape) # (2000, 3)
plt.figure(figsize=(10,2))
plt.plot(S,'-')
plt.xlabel("Time") 
plt.ylabel("Signal")
plt.show()
# Here we cannot observe the original signals, but only an additive mixture of all three of them.
# We want to recover the decomposition of the mixed signal into the original components. 
A=np.random.RandomState(0).uniform(size=(100,3))
print(A.max(),A.min()) # Value between 0 and 1
print(A.T.shape) # Transpose (3, 100)
X=np.dot(S,A.T)
print(X.shape) # (2000, 100)
# Now use nmf to recover original signal from X
nmf=NMF(n_components=3,random_state=42)
S_=nmf.fit_transform(X) 
print(S_.shape)  # (2000, 3)
# Now use PCA to recover original signal from X
pca=PCA(n_components=3)
H=pca.fit_transform(X)
print(H.shape) # (2000, 3)
# Now plot the graph of these 4 signals of Original,X,S_,H
models = [X, S, S_, H]
names = ['Observations (first three measurements)','True sources','NMF recovered signals','PCA recovered signals']
# gridspec_kw is used for adjusting the distance between an axes plot and its title
fig, axes = plt.subplots(4, figsize=(10, 4), gridspec_kw={'hspace': .5} ,subplot_kw={'xticks': (), 'yticks': ()})
for model, name, ax in zip(models, names, axes):    
    ax.set_title(name)
    ax.plot(model[:,:3],'-')
plt.show()
# NMF did a reasonable job of discovering the original sources, while PCA failed and used the first component to explain the majority of the variation in the data.
# Keep in mind that the components produced by NMF have no natural ordering.
# In this example, the ordering of the NMF components is the same as in the original signal (see the shading of the three curves), but this is purely accidental.

# Reconstruction of Faces using nmf :-
# In nmf also, X_train_nmf cannot be used to reconstruct the image because it has only 15 columns while there should be 5655 for image shape 87 X 65
# X_test_nmf (3023,15) and nmf.components_ (15,5655) gives the array for reconstruction of shape (3023,5655)
nmf=NMF(n_components=15).fit(X_train) # fitting X_train or X_test doesn't make any difference
rcon_nmf=np.dot(nmf.transform(X_test),nmf.components_) # Here we can also use X_test_nmf
print(rcon_nmf.shape) # (516,5655)
fig,axes=plt.subplots(3,5,figsize=(10,8),subplot_kw={'xticks': (), 'yticks': ()})
for ax,r in zip(axes.ravel(),rcon_nmf):
    ax.imshow(r.reshape(image_shape))
plt.show()

n=np.dot(nmf.transform(X_test),nmf.components_)
print(n[0],'\n',X_test[0])
