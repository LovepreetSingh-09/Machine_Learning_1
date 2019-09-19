# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 02:04:26 2019

@author: user
"""
# Uncertainity Estimates in Classifiers

import graphviz
import mglearn
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
from IPython.display import display
from sklearn.model_selection import train_test_split

# Often, you are not only interested in which class a classifier predicts for a certain test point, but also how certain it is that this is the right class. 
#This is known as uncertainity
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.datasets import make_circles
X,y=make_circles(noise=0.25,factor=0.5,random_state=1)
print(y)
# Assing value to the classifier y, 0 = blue , 1 = red
#we rename the classes "blue" and "red" for illustration purposes
y_named=np.array(['blue','red'])[y]
print(y_named)
print(len(y_named),len(X),len(y))
# we can call train_test_split with arbitrarily many arrays;
# all will be split in a consistent manner
X_train,X_test,y_train_named,y_test_named,y_train,y_test=train_test_split(X,y_named,y,random_state=0)
gbrt=GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train,y_train_named)

# Decision Function
# Provides Certainity score
# Class with the max certainity score will be the result
print("X_test.shape: {}".format(X_test.shape)) 
print("Decision function shape: {}".format(gbrt.decision_function(X_test).shape)) 
# This value encodes how strongly the model believes a data point to belong to the “positive” class, in this case class 1.
# Positive values indicate a preference for the positive class, and negative values indicate a preference for the “negative” (other) class
print("Decision function shape: {}".format(gbrt.decision_function(X_test)[:6]))
# This gives the values in true/false instead of negative/positive values
# True=Class1  and False=Class0
print("Thresholded decision function:\n{}".format(gbrt.decision_function(X_test)>0))
print("Predictions:\n{}".format(gbrt.predict(X_test)))  
# For binary classification, the “negative” class is always the first entry of the classes_ attribute, and the “positive” class is the second entry of classes_.
# So if you want to fully recover the output of predict, you need to make use of the classes_ attribute
# make the boolean True/False into 0 and 1
greater_zero = (gbrt.decision_function(X_test) > 0).astype(int)
# use 0 and 1 as indices into classes_ 
pred = gbrt.classes_[greater_zero]
# pred is the same as the output of gbrt.predict 
print("pred is equal to predictions: {}".format(np.all(pred == gbrt.predict(X_test))))
# The range of decision_function can be arbitrary, and depends on the data and the model parameters
decision_function = gbrt.decision_function(X_test)
print("Decision function minimum: {:.2f} maximum: {:.2f}".format(np.min(decision_function), np.max(decision_function)))

# we plot the decision_function for all points in the 2 D plane using a color coding, next to a visualization of the decision boundary.
# We show training points as circles and test data as triangles 
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
mglearn.tools.plot_2d_separator(gbrt, X, ax=axes[0], alpha=.4,fill=True, cm=mglearn.cm2) # makes boundary separator for 1st plot
scores_image = mglearn.tools.plot_2d_scores(gbrt, X, ax=axes[1],alpha=.4, cm=mglearn.ReBl) # makes boundary separation for 2nd plot
for ax in axes:
    # plot training and test points
    mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test,markers='^', ax=ax)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train,markers='o', ax=ax)
    ax.set_xlabel("Feature 0")     
    ax.set_ylabel("Feature 1")
cbar = plt.colorbar(scores_image, ax=axes.tolist()) # makes a colorbar at the right most side of the whole plot
print(axes.tolist())
axes[0].legend(["Test class 0", "Test class 1", "Train class 0","Train class 1"], ncol=4, loc=(.1, 1.1))
plt.show()

# Predicting probabilities
# The output of predict_proba is a probability for each class
print("Shape of probabilities: {}".format(gbrt.predict_proba(X_test).shape)) 
# The first entry in each row is the estimated probability of the first class, and the second entry is the estimated probability of the second class.
# The output of predict_proba is always between 0 and 1, and the sum of the entries for both classes is always 1
# show the first few entries of predict_proba 
print("Predicted probabilities:\n{}".format(gbrt.predict_proba(X_test[:6])))
# A model is called calibrated if the reported uncertainty actually matches how correct it is—in a calibrated model, a prediction made with 70% certainty would be correct 70% of the time.
# 2 plots for probabilities
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
mglearn.tools.plot_2d_separator(gbrt, X, ax=axes[0], alpha=.4,fill=True, cm=mglearn.cm2) # makes boundary separator for 1st plot
scores_image = mglearn.tools.plot_2d_scores(gbrt, X, ax=axes[1],alpha=.4, cm=mglearn.ReBl,function='predict_proba') # makes boundary separation for 2nd plot
for ax in axes:
    # plot training and test points
    mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test,markers='^', ax=ax)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train,markers='o', ax=ax)
    ax.set_xlabel("Feature 0")     
    ax.set_ylabel("Feature 1")
cbar = plt.colorbar(scores_image, ax=axes.tolist()) # makes a colorbar at the right most side of the whole plot
print(axes.tolist())
axes[0].legend(["Test class 0", "Test class 1", "Train class 0","Train class 1"], ncol=4, loc=(.1, 1.1))
plt.show()
# The boundaries in this plot are much more well-defined, and the small areas of uncertainty are clearly visible

# Uncertainity in multiclass cassificatiom
from sklearn.datasets import load_iris
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
gbrt = GradientBoostingClassifier(learning_rate=0.01, random_state=0) 
gbrt.fit(X_train, y_train)
print("Decision function shape: {}".format(gbrt.decision_function(X_test).shape))
# plot the first few entries of the decision function
print("Decision function:\n{}".format(gbrt.decision_function(X_test)[:6]))
# gives the position of the max value column wise and the output of both of the below code is same
print("Argmax of decision function:\n{}".format(np.argmax(gbrt.decision_function(X_test), axis=1))) 
print("Predictions:\n{}".format(gbrt.predict(X_test)))  

# show the first few entries of predict_proba
print("Predicted probabilities:\n{}".format(gbrt.predict_proba(X_test)[:6]))
# show that sums across rows are one
print("Sums: {}".format(gbrt.predict_proba(X_test)[:6].sum(axis=1))) 
# gives the position of the max value column wise and the output of both of the below code is same
print("Argmax of probabilities:\n{}".format(np.argmax(gbrt.predict_proba(X_test), axis=1))) 
print("Predictions:\n{}".format(gbrt.predict(X_test)))  

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
# represent each target by its class name in the iris dataset
named_target = iris.target_names[y_train] 
logreg.fit(X_train, named_target)
print("unique classes in training data: {}".format(logreg.classes_)) 
print("predictions: {}".format(logreg.predict(X_test)[:10])) 
argmax_dec_func = np.argmax(logreg.decision_function(X_test), axis=1) 
print("argmax of decision function: {}".format(argmax_dec_func[:10]))
print("argmax combined with classes_: {}".format(logreg.classes_[argmax_dec_func][:10])) 