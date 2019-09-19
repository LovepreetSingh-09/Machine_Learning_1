# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 13:58:17 2019

@author: user
"""

# As the PCA model is based on pixels, the alignment of the face (the position of eyes, chin, and nose) and the lighting both have a strong influence on how similar two images are in their pixel representation.

#import graphviz
import mglearn
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Another application of PCA that we mentioned earlier is feature extraction.
# The idea behind feature extraction is that it is possible to find a representation of your data that is better suited to analysis than the raw representation you were given.
# A great example of an application where feature extraction is helpful is with images. Images are made up of pixels, usually stored as red, green, and blue (RGB) intensities.
# Objects in images are usually made up of thousands of pixels, and only together are they meaningful.
from sklearn.datasets import fetch_lfw_people
people=fetch_lfw_people(min_faces_per_person=20,resize=0.7) # resize affects the pixels more the resize more will be the pixels
print(people.keys()) # (['data', 'images', 'target', 'target_names', 'DESCR']) 
print(people.target) # 1D array of shape(3023, ) of no. from 0 to 61 represent the classor target_name
print(people.target.shape) # (3023,)
print(people.target_names.shape) # (62,)
print(people.images.shape) # (3023, 87,65) means 3023 images with 87 X 65 pixels
print(people.data.shape) # (3023, 5655)  rows represent respective image while columns are data for the pixels 87 X 65 = 5655
# print(people.data)
print(people.images)
image_shape=people.images[0].shape
print(image_shape) # represent pixels (87,65) 87 is on the vertical y_axis and 65 is on horizontal x-axis of a single image
fig,axes=plt.subplots(2,5,figsize=(15,8),subplot_kw={'xticks':(),'yticks':()}) # subplot_kw is for disabling x and y ticks
print(axes.ravel()) # creates the  axis objets
for target,image,ax in zip(people.target,people.images,axes.ravel()):
    ax.imshow(image)
    ax.set_title(people.target_names[target])
    # we can also use the following for disabling x and y ticks
    ax.set_xticks(())
    ax.set_yticks(())
plt.show()

# counts how many times each target appears which represents no. of images of each person or class
counts=np.bincount(people.target) 
print(counts)
for count,name in zip(counts,people.target_names):
    print('{0:25} {1:5}'.format(name,count),end='            ')
   
# To make the data less skewed, we will only take up to 50 images of each person(otherwise, the feature extraction would be overwhelmed by the likelihood of George W. Bush)
mask=np.zeros(people.target.shape,dtype=np.bool)
print(mask) # makes 1D array of 3023 elements in boolean (True/False) format. Here, all False because of np.zeros
print(np.unique(people.target)) # no. from 0 to 61
for target in np.unique(people.target):
    # assigns True to the first 50 values of target remaining will be False
    mask[np.where(people.target==target)[0][:50]]=1
    # Get the values of index of each class present in the people.target(3023,) and makes 62 1D arrays of each class containing those index values and then limiting those values upto 50  
    # Then the value of those indexes present in 62 1D arrays are asigned 1 or True and indexes not in these arrays will remain False 
    print((np.where(people.target==target)[0][:50]).shape) # 62 1D arrays of maximum shape (50,)
    # 0 represent the no. of index of 62 1D arrays
print(people.target,'\n',mask) # mask represent the 0 or False to the last indexes which represents the same class over 50 times
X_people=people.data[mask]
print(X_people.shape) # (2063, 5655)
print(X_people.max()) # 255.0
y_people=people.target[mask]
print(y_people.shape) # (2063,)
# X_people and y_people are the new data and target obtained from the dataset
# scale the grayscale values to be between 0 and 1
# make value from 0 and 255 for better numeric stability 
X_people = X_people / 255.0
print(X_people.min()) # 0.0
print(X_people.max()) # 1.0

# Using KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
# build a KNeighborsClassifier using one neighbor 
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print(X_train.shape) # (1547, 5655)
print("Test set score of 1-nn: {:.2f}".format(knn.score(X_test, y_test))) #  0.23

# Using PCA with whitening
# Whitening rescales the principal components to have the same scale
# Whitening actually means to use the StandardScaler after transformatiom in the PCA
# center Panel is a circle instead of an ellipse
mglearn.plots.plot_pca_whitening()
plt.show()
pca=PCA(n_components=100,whiten=True,random_state=0).fit(X_train)
X_train_pca=pca.transform(X_train)
X_test_pca=pca.transform(X_test)
print('Shape of X_train_pca : ',X_train_pca.shape) #  (1547, 100) Hence, dimensional reduction by reducing the no. of columns
print('Shape of pca.components_ : ',pca.components_.shape) #  (100, 5655)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca, y_train)
print("Test set accuracy using PCA Whitening : {:.2f}".format(knn.score(X_test_pca, y_test))) # 0.31

print(pca.components_[0].shape) # (5655,) 
# Reshape the 1st pca.components_[0] into image shape or pixels 87 X 65
# Converting 1D array of a single pca.components_[0] into 2D array because 5655 = 87 X 65
print(pca.components_[0].reshape(image_shape).shape)
fig,axes=plt.subplots(3,5,figsize=(15,8),subplot_kw={'xticks':(),'yticks':()})
# components correspond to directions in the input space. 
# The input space here is 87×65-pixel grayscale images, so directions within this space are also 87 ×65-pixel grayscale images.
for i,(component,ax) in enumerate(zip(pca.components_,axes.ravel())):
    ax.imshow(component.reshape(image_shape))
    ax.set_title('{} componet'.format(i+1))
plt.show()
# The first component seems to mostly encode the contrast between the face and the background, the second component encodes differences in lighting between the right and the left half of the face, and so on. 
# As the PCA model is based on pixels, the alignment of the face (the position of eyes, chin, and nose) and the lighting both have a strong influence on how similar two images are in their pixel representation.
# But humans rate similarity of images based on age,gender,hair-style etc rather than lighting, alignments etc. 
# Hence, algorithms often interpret data (particularly visual data, such as images) quite differently from how a human would.

# ŷ = w[0] * x[0] + w[1] * x[1] + ... 
# y = final data point , x0, x1, and so = coefficients of the principal components for this data point
# w[0],w[1] and so on components

# Reconstructing three face images using increasing numbers of principal components
# The return to the original feature space can be done using the inverse_transform method.
# Here, we visualize the reconstruction of some faces using 10, 50, 100, or 500 components.
mglearn.plots.plot_pca_faces(X_train,X_test,image_shape)
plt.show()
# You can see that when we use only the first 10 principal components, only the essence of the picture, like the face orientation and lighting, is captured.
# By using more and more principal components, more and more details in the image are preserved.
# This corresponds to extending the sum to include more and more terms.
# Using as many components as there are pixels would mean that we would not discard any information after the rotation, and we would reconstruct the image perfectly.

# Scatter Plot of PCA
mglearn.discrete_scatter(X_train_pca[:, 0], X_train_pca[:, 1], y_train)
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.show()
# PCA only captures very rough characteristics of the faces.

# Reconstruction of Faces :-
# We cannot reconstruct the images from the X_train_pca because they are (3023,100) and 100 component columns which are actually features cannot be reshape into 87 X 65
# For reconstruction of images, we can use the pca.inverse_transform method
# Transformation for the faces by reducing the data to only some principal components and then rotating back into the original space after dropping some other components.
# This return to the original feature space can be done using the inverse_transform method.
fig,axes=plt.subplots(3,5,figsize=(15,8),subplot_kw={'xticks':(),'yticks':()})
rcon_pca=pca.inverse_transform(pca.transform(X_test)) # here, we can also use X_train_pca to rconstruct those faces after dropping some components
# rcon_pca=np.dot(pca.transform(X_test),pca.components_) # This method can also be used but this is about the patterns which is used in nmf
for i,(r,ax) in enumerate(zip(rcon_pca,axes.ravel())):
    ax.imshow(r.reshape(image_shape))
plt.show()

# Inverse Transform also tries to recover the X_train values by dropping some components.
# Although the value willl not be the same but they are little bit close
print(X_train)
print(pca.inverse_transform(X_train_pca))

