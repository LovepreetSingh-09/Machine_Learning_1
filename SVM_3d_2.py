# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 00:26:27 2019

@author: user
"""

# Kernelized Support Vector Machines 2
import graphviz
import mglearn
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
from IPython.display import display
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
svc = SVC() 
svc.fit(X_train, y_train)
print("Accuracy on training set: {:.2f}".format(svc.score(X_train, y_train))) # Accuracy on training set: 1.00
print("Accuracy on test set: {:.2f}".format(svc.score(X_test, y_test))) # Accuracy on test set: 0.63
# Hence The model is very much overfit
# This is because the SVC reguires the features to be on a very similar scale

# Now look at the magnitude of each feature in logspace with maximum and minimum values 
plt.boxplot(X_train,manage_xticks=False)
plt.yscale('symlog')
plt.xlabel("Feature index") 
plt.ylabel("Feature magnitude")
plt.show()
# From this plot we can determine that features in the Breast Cancer dataset are of completely different orders of magnitude. 

# Preprocessing Of Data in SVM
# Now we need to rescale each farture to be it on the same scale
# A common rescaling method for kernel SVMs is to scale the data such that all features are between 0 and 1. 
min_on_training=X_train.min(axis=0) # compute the minimum value per feature on the training set
print(X_train,'\n',X_train.max(axis=0),'\n',min_on_training)
# compute the range of each feature (max - min) on the training set 
# After subtracting the min from every value of each feature, we get the maximum of that feature
range_on_training=(X_train-min_on_training).max(axis=0)
print(range_on_training)
# subtract the min, and divide by range from the original values of X
# afterward, min=0 and max=1 for each feature
X_train_scaled=(X_train-min_on_training)/range_on_training
print(X_train_scaled)
X_train_scaled = (X_train - min_on_training) / range_on_training 
print("Minimum for each feature\n{}".format(X_train_scaled.min(axis=0))) 
print("Maximum for each feature\n {}".format(X_train_scaled.max(axis=0)))

# Now use the same values of X_train on X_test for rescaling
X_test_scaled = (X_test - min_on_training) / range_on_training 
# For test_scaled the minimum and max won't be 0 and 1 , they will vary from negative number to more than 1
print("Minimum for each feature\n{}".format(X_test_scaled.min(axis=0))) 
print("Maximum for each feature\n {}".format(X_test_scaled.max(axis=0)))
svc = SVC()
svc.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(svc.score(X_train_scaled, y_train))) # Accuracy on training set: 0.948
print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))  # Accuracy on training set: 0.948
# As you can see the mode's accuracy is very good after rescalingbut it is slightly underfit
# So we can try increasing either C or gamma to fit a more complex model. 
svc = SVC(C=1000)
svc.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(svc.score(X_train_scaled, y_train))) # Accuracy on training set: 0.988
print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))      # Accuracy on test set: 0.972
# Now by making a complex model by C , the acuracy and performance got better and on par with all other algorithms