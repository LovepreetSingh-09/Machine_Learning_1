# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 01:13:21 2019

@author: user
"""
import mglearn
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split

# Binary Classification :-

# ŷ = w[0] * x[0] + w[1] * x[1] + ... + w[p] * x[p] + b > 0
# If the function is smaller than zero, we predict the class –1; if it is larger than zero, we predict the class +1.
# The two most common linear classification algorithms are 1.) logistic regression, implemented in linear_model.LogisticRegression ; Despite its Regression in name It is a classification algorithm
# 2.) linear support vector machines (linear SVMs), implemented in svm.LinearSVC (SVC stands for support vector classifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
X,y=mglearn.datasets.make_forge()
#X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)
fig,axes=plt.subplots(1,2,figsize=(10,5))
for model,ax in zip([LinearSVC(),LogisticRegression()],axes):
    clf=model.fit(X,y)
    mglearn.plots.plot_2d_separator(clf,X,fill=False,eps=0.5,alpha=0.8,ax=ax)
    mglearn.discrete_scatter(X[:,0],X[:,1],y,ax=ax)
    # Below always use double underscore
    ax.set_title('{}'.format(clf.__class__.__name__))
    ax.set_xlabel('Feature 0')
axes[0].set_ylabel('Feature 1')
axes[0].legend()
plt.show()

# Here in linear classification regularization factor is defined by the term C 
# Higher value of C means lesser regularization . So, it is opposite to that of in linear regression
# Using low values of C will cause the algorithms to try to adjust to the “majority” of data points, 
# while using a higher value of C stresses the importance that each individual data point be classified correctly.
mglearn.plots.plot_linear_svc_regularization()
plt.show()
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=42)
logreg=LogisticRegression().fit(X_train,y_train)
# Training set score: 0.955 and Test set score: 0.958
# training and test set performance are very close, it is likely that we are underfitting.
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))
# For more accuracy and to avoid underfitting Use C=10
logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
# Training set score: 0.972 and Test set score: 0.965 So, at higher value of C and lower regularization accuracy increases
print("Training set score: {:.3f}".format(logreg100.score(X_train, y_train))) 
print("Test set score: {:.3f}".format(logreg100.score(X_test, y_test)))
# Now use C=0.01
logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
# Training set score: 0.934 and Test set score: 0.930 So at higher regularization by using lower value of C the accuracy is decreased
print("Training set score: {:.3f}".format(logreg001.score(X_train, y_train))) 
print("Test set score: {:.3f}".format(logreg001.score(X_test, y_test))) 

print(range(cancer.data.shape[1])) # 1 = no. of columns and 0 = no. of rows
print(cancer.data.shape) # (569, 30)
print(range(cancer.feature_names.shape[0])) # It is equal to the above line of range

# From the plot it is clear that lesser value of C is more regularized
plt.plot(logreg.coef_.T, 'o', label="C=1") 
plt.plot(logreg100.coef_.T, '^', label="C=100") 
plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim(-5, 5) 
plt.xlabel("Feature") 
plt.ylabel("Coefficient magnitude") 
plt.legend()
plt.show()

# Now make the similar graph with L1 regularization
# for  this graph also the accuracy of C=100 is the maximum and C=0.001 is minimum even the minimum=0.91 and 0.92
for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):    
    lr_l1 = LogisticRegression(C=C, penalty="l1").fit(X_train, y_train)    
    print("Training accuracy of l1 logreg with C={:.3f}: {:.2f}".format( C, lr_l1.score(X_train, y_train)))
    print("Test accuracy of l1 logreg with C={:.3f}: {:.2f}".format( C, lr_l1.score(X_test, y_test)))
    plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.xlabel("Feature") 
plt.ylabel("Coefficient magnitude")
plt.ylim(-5, 5) 
plt.legend(loc=3) 
plt.show()

# Multi-Class Classification

# In the one-vs.-rest approach, a binary model is learned for each class that tries to separate that class from all of the other classes, resulting in as many binary models as there are classes. 
# w[0] * x[0] + w[1] * x[1] + ... + w[p] * x[p] + b
from sklearn.datasets import make_blobs
X, y = make_blobs(random_state=42) 
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0") 
plt.ylabel("Feature 1") 
plt.legend(["Class 0", "Class 1", "Class 2"])
plt.show()

linear_svm=LinearSVC().fit(X,y)
print('Coefficient or slope are :',linear_svm.coef_)
# Coefficient or slope shape is : (3, 2)  means rows are the classes and columns are features (here =2) So, every class has its own slope for each feature
print('Coefficient or slope shape is :',linear_svm.coef_.shape)
print('Intercepts are :',linear_svm.intercept_)
# Intercepts are : [-1.0774537   0.13140278 -0.08604839] Intercepts are single for each class
print('Intercepts shape is :',linear_svm.intercept_.shape)

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line=np.linspace(-15,15)
for coef,intercept,color in zip(linear_svm.coef_,linear_svm.intercept_,mglearn.cm3.colors):
    plt.plot(line,-(line*coef[0]+intercept)/coef[1],color=color) # Here coef[0] = coefficient of first feature and coef[1] = coefficient of second feature
    # The area of a specific class is the area made by dividing the area into half where the its own line cuts the other 2 classes lines in that class' region
    print(coef[0],'\n',coef[1])
plt.ylim(-10, 15) 
plt.xlim(-10, 8) 
plt.xlabel("Feature 0") 
plt.ylabel("Feature 1")
plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1','Line class 2'], loc=(1.01, 0.3))
plt.show()

mglearn.plots.plot_2d_classification(linear_svm,X,fill=True,alpha=0.7)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line=np.linspace(-15,15)
for coef,intercept,color in zip(linear_svm.coef_,linear_svm.intercept_,mglearn.cm3.colors):
    plt.plot(line,-(line*coef[0]+intercept)/coef[1],color=color) # Here coef[0] = coefficient of first feature and coef[1] = coefficient of second feature
    # The area of a specific class is the area made by dividing the area into half where the its own line cuts the other 2 classes lines in that class' region
    # Now the triangle remained between those three lines is equally divided into 3 parts  
    print(coef[0],'\n',coef[1])
plt.ylim(-10, 15) 
plt.xlim(-10, 8) 
plt.xlabel("Feature 0") 
plt.ylabel("Feature 1")
plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1','Line class 2'], loc=(1.01, 0.3))
plt.show()

logreg = LogisticRegression().fit(X_train, y_train)
# This concatenation of method calls (here __init__ and then fit) is known as method chaining. 


