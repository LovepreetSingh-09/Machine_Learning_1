# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 00:06:46 2019

@author: user
"""

# Kernelized Support Vector Machines
import graphviz
import mglearn
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
from IPython.display import display
from sklearn.model_selection import train_test_split

from sklearn.datasets import make_blobs
X, y = make_blobs(centers=4, random_state=8) 
# In the dataset there are 4 classes but we made it for 2 classes by reemainder form of y
y = y % 2
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0") 
plt.ylabel("Feature 1")
plt.show()

# Following graph makes a wrong decision boundary
from sklearn.svm import LinearSVC
linear_svm=LinearSVC().fit(X,y)
mglearn.plots.plot_2d_separator(linear_svm,X)
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.xlabel('feature 0')
plt.ylabel('feature 1')
plt.show()

# Make a new datasets of X having one more feature which is the sq. of feature 1
# hp.stack adds a new column of the new feature
X_new=np.hstack([X,X[:,1:]**2])
from mpl_toolkits.mplot3d import Axes3D,axes3d
figure=plt.figure()
# Visualize in 3d elev=rotation around x-axis while azim=ratation around y=axis
# - sign in elev and azim rotate the 3d plot oppositely elev rotate opposite vertically while azim horizontally
ax=Axes3D(figure,elev=-152,azim=-26)
# Now mask represent the class 0 and ~mask represent class 1
mask=y==0
ax.scatter(X_new[mask,0],X_new[mask,1],X_new[mask,2],c='b',cmap=mglearn.cm3,s=60,edgecolor='g')
ax.scatter(X_new[~mask,0],X_new[~mask,1],X_new[~mask,2],c='r',cmap=mglearn.cm3,s=60,marker='v',edgecolor='k')
ax.set_xlabel('Feature 0')
ax.set_ylabel('Feature 1')
ax.set_zlabel('Feature 2(X[:,1]**2)')
plt.show()

# Separation of 2 classes
linear_svm_3d=LinearSVC().fit(X_new,y)
# ravel() converts the 2d array into 1D array. Here, originally the slopes was in 2D array of just 1 row and 3 columns
coef,intercept=linear_svm_3d.coef_.ravel(),linear_svm_3d.intercept_
print(coef,intercept)
figure=plt.figure()
ax=Axes3D(figure,elev=-152,azim=-26)
print(X_new[:,0].min())
# Now xx and yy makes an array having elements starts from less than 2 min of X_new to more than 2 max of X_new   
xx=np.linspace(X_new[:,0].min()-2,X_new[:,0].max()+2,50)
yy=np.linspace(X_new[:,1].min()-2,X_new[:,1].max()+2,50)
print(xx,'\n',yy)
XX,YY=np.meshgrid(xx,yy)
# Both XX and YY is a 2D array of size(50,50)
# Meshgrid actually makes a 2D array made from an array by increasing the row to a no. of times the columns is in that array
# Like xx has a 1 row of 50 columns and XX has 50 rows and 50 columns. All the 50 rows are same as the 1st row 
# Then Meshgrid converts the 2nd array of bracket from 1 row to 1 column(Transpose) and then repeat that column no. of times as of rows of transpose 
print(XX,'\n',YY)
ZZ=(coef[0]*XX+coef[1]*YY+intercept)/-coef[2]
print(ZZ)
ax.plot_surface(XX,YY,ZZ,rstride=8,cstride=8,alpha=0.3)
ax.scatter(X_new[mask,0],X_new[mask,1],X_new[mask,2],c='g',cmap=mglearn.cm2,s=60,edgecolor='k')
ax.scatter(X_new[~mask,0],X_new[~mask,1],X_new[~mask,2],c='r',cmap=mglearn.cm2,s=60,marker='^',edgecolor='k')
ax.set_xlabel('Feature 0')
ax.set_ylabel('Feature 1')
ax.set_zlabel('Feature 2(X[:,1]**2)')
plt.show()

ZZ=YY**2
print(XX.ravel(),YY.ravel(),ZZ.ravel())
# Here XX.ravel() converts the 2D array of shape(50,50) into a 1D array of 2500 elements
# Then np.c_ makes 2D array of 3 columns (1st column=XX, 2nd=YY, 3rd=ZZ)
print(np.c_[XX.ravel(),YY.ravel(),ZZ.ravel()].shape)
print(np.c_[XX.ravel(),YY.ravel(),ZZ.ravel()])
# dec = decision boundary
dec=linear_svm_3d.decision_function(np.c_[XX.ravel(),YY.ravel(),ZZ.ravel()])
# dec= 1D array of 2500 elements
# Now dec.reshape(XX.shape) cajnges the shape to (50,50)
print(dec.reshape(XX.shape).shape)
print(dec)
levels=[dec.min(),0,dec.max()]
print(levels)
plt.contourf(XX,YY,dec.reshape(XX.shape),levels=[dec.min(),0,dec.max()],cmap=mglearn.cm2,alpha=0.5)
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.xlabel("Feature 0") 
plt.ylabel("Feature 1")
plt.legend()
plt.show()

# there is a clever mathematical trick that allows us to learn a classifier in a higher-dimensional space without actually computing the new, possibly very large representation. This is known as the kernel trick
# There are 2 ways:- 1.) Polynomial Kernel (computes all possible polynomials up to a certain degree of the original features (like feature1 ** 2 * feature2 ** 5))
# 2.) RBF(radial basis function) or Gaussian Kernel, it considers all possible polynomials of all degrees, but the importance of the features decreases for higher degrees.
# Support Vectors :- subset of the training points matter for defining the decision boundary: the ones that lie on the border between the classes. 
# A classification decision is made based on the distances to the support vector, and the importance of the support vectors that was learned during training (stored in the dual_coef_ attribute of SVC).
# The distance between data points is measured by the Gaussian kernel:
# krbf(x1, x2) = exp (–ɣǁx1 - x2ǁ2)
# x1 and x2 are data points, ǁ x1 - x2 ǁ denotes Euclidean distance, and ɣ ( gamma ) is a parameter that controls the width of the Gaussian kernel.
from sklearn.svm import SVC
X,y=mglearn.tools.make_handcrafted_dataset() # 2 dimensional or feature 2 class dataset
svm=SVC(kernel='rbf',C=10,gamma=0.1).fit(X,y)
mglearn.plots.plot_2d_separator(svm,X,eps=0.5)
mglearn.discrete_scatter(X[:,0],X[:,1],y)
# plot Support vectors
sv=svm.support_vectors_
print(sv)
# # class labels of support vectors are given by the sign of the dual coefficients
sv_labels=svm.dual_coef_.ravel()>0
print(sv_labels)
print(svm.dual_coef_.ravel())
mglearn.discrete_scatter(sv[:,0],sv[:,1],sv_labels,s=15,markeredgewidth=3)
plt.xlabel("Feature 0") 
plt.ylabel("Feature 1")
plt.show()

# gamma = inverse of width of gaussian kernel
# gamma parameter determines how far the influence of a single training example reaches, with low values meaning corresponding to a far reach, and high values to a limited reach.
# small gamma means larger radius and the wider decision boundary and large gamma makes a very complex model by considering the data points in a tight boindary
# The C parameter is a regularization parameter, similar to that used in the linear models. It limits the importance of each point (or more precisely, their dual_coef_).
# Small C means a very restrictive model and no variation of the line making a simple model whilw larger value make a bended or non-linear boundary to make decision boundary complex
fig,axes=plt.subplots(3,3,figsize=(15,10))
for ax,C in zip(axes,[-1,0,3]): # log(-1)=0.1 log(3)=1000 by the base of 10
    for a,gamma in zip(ax,range(-1,2)): # range(-1,2)=[-1,0,1]
        mglearn.plots.plot_svm(log_C=C,log_gamma=gamma,ax=a) # log(0)=1
axes[0,0].legend(["class 0", "class 1", "sv class 0", "sv class 1"],ncol=4, loc=(.9, 1.2))
plt.show()









