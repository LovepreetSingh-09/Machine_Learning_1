# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 23:37:15 2019

@author: user
"""

# Linear Models for regression
import mglearn
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split

# ŷ = w[0] * x[0] + w[1] * x[1] + ... + w[p] * x[p] + b
#  Where y= prediction the model makes ; p+1= No. of totsl features ; x[p]= denotes the features of a single data point
# w= slope or coefficient ; b= y-intercept

#For a dataset with a single feature, this is:
#ŷ = w[0] * x[0] + b

mglearn.plots.plot_linear_regression_wave() #It also gives the slope and the y-intercept values with the graph or plot
plt.show()
# regression dataset
X,y=mglearn.datasets.make_wave(n_samples=60)
from sklearn.linear_model import LinearRegression
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)
lre=LinearRegression().fit(X_train,y_train)
print('The slope or co-efficient is : ',lre.coef_)
print('The y-intercept is : ',lre.intercept_)
#The Train_score or accuracy is :  0.6700890315075756 and The Test_score or accuracy is :  0.65933685968637
#Hence the linear regression doesn't work well on small datsets both on training and test datasets
print('The Train_score or accuracy is : ',lre.score(X_train,y_train))
print('The Test_score or accuracy is : ',lre.score(X_test,y_test))


X,y=mglearn.datasets.load_extended_boston()
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
lre=LinearRegression().fit(X_train,y_train)
print('The slope or co-efficient is : ',lre.coef_) # Slope or coefficient is different for every feature and it is stored in an array
print('The y-intercept is : ',lre.intercept_) # y-intercept is just a single float no. for all of the features
#The Train_score or accuracy is :  0.9520519609032729 and The Test_score or accuracy is :  0.6074721959665752
# Linear regression now has done overfitting which arises in small no. of data available which means the predictions work well on training data but not on test data
print('The Train_score or accuracy is : ',lre.score(X_train,y_train))
print('The Test_score or accuracy is : ',lre.score(X_test,y_test))
 
# To avoid this overfitting we use Ridge Regression
# Regularization means explicitly restricting a model to avoid overfitting.
#In simple words Regularization means w or slope should be close to zero while predicting well
# Ridge uses L2 regularization
from sklearn.linear_model import Ridge
ridge=Ridge().fit(X_train,y_train)
# The Train_score or accuracy is :  0.885796658517094 and The Test_score or accuracy is :  0.7527683481744754
# Here Training score is lower but test score is higher than LinearRegression Because Ridge is restricted model so lesser probability of overfitting
print('The Train_score or accuracy is : ',ridge.score(X_train,y_train))
print('The Test_score or accuracy is : ',ridge.score(X_test,y_test))
 # We can set the restriction by using alpha and by default alpha=1
 # Increasing alpha forces coefficients to move more toward zero, which decreases training set performance and it means we are more restrictive
ridge10=Ridge(alpha=10).fit(X_train,y_train)
# The Train_score or accuracy is :  0.7882787115369614 and The Test_score or accuracy is :  0.6359411489177311
print('The Train_score or accuracy by alpha=10 is : ',ridge10.score(X_train,y_train))
print('The Test_score or accuracy by alph=10 is : ',ridge10.score(X_test,y_test))
# Decreasing alpha allows the coefficients to be less restricted and coefficients are barely restricted and we end up with a model that resembles LinearRegression
ridge01=Ridge(alpha=0.1).fit(X_train,y_train)
# The Train_score or accuracy is :  0.9282273685001986 and The Test_score or accuracy is :  0.7722067936479804
# Hence by decreasing the alpha or restriction we get a better score
print('The Train_score or accuracy by alpha=0.1 is : ',ridge01.score(X_train,y_train))
print('The Test_score or accuracy by alpha=0.1 is : ',ridge01.score(X_test,y_test))
# We can describe the variability by using different values of alpha
plt.plot(ridge.coef_,'s',label='Ridge alpha=1')
plt.plot(ridge10.coef_,'^',label='Ridge alpha=10')
plt.plot(ridge01.coef_,'v',label='Ridge alpha=0.1')
plt.plot(lre.coef_,'o',label='Linear regression')
# X-axis represent features
plt.xlabel('Coefficient Index')
# y-axis represent slope or coefficient
plt.ylabel('Coefficient Magnitude')
# In hlines(start at x-axis,start at y-axis,length of line)
plt.hlines(0,0,len(ridge.coef_))
plt.ylim(-25,25)
plt.legend()
plt.show()

mglearn.plots.plot_ridge_n_samples()
# This graph shows the variability of training and test score of both ridge and Linear regression But here alpha=1 by default
plt.show()

# An alternative to Ridge for regularizing linear regression is Lasso
# Lasso works on L1 regularization which makes some coefficients are exactly zero.Thus, some features are fully ignored
from sklearn.linear_model import Lasso
lasso=Lasso().fit(X_train,y_train)
# lasso does underfitting normally
# The Train_score or accuracy by lasso is :  0.29323768991114607 and The Test_score or accuracy by lasso is :  0.20937503255272294
# Hence It does very bad on train and test data because of underfitting which arises due to very simple model 
print('The Train_score or accuracy by lasso is : ',lasso.score(X_train,y_train))
print('The Test_score or accuracy by lasso is : ',lasso.score(X_test,y_test))
# The features used by Lasso are :  4  So, It has ignoed the remaining 101 features
print('The features used by Lasso are : ',np.sum(lasso.coef_!=0))

# Like ridge Lasso also has regularization parameter alpha
# While defining alpha ,we also need to increase the default setting of max_iter (the maximum number of iterations to run)
lasso001=Lasso(alpha=0.01,max_iter=100000).fit(X_train,y_train)
# The Train_score or accuracy by lasso by alpha=0.01 is :  0.8962226511086497 and The Test_score or accuracy by lasso by alpha=0.01 is :  0.7656571174549983
# Now by reducing the regularization the train and test score has increased
print('The Train_score or accuracy by lasso by alpha=0.01 is : ',lasso001.score(X_train,y_train))
print('The Test_score or accuracy by lasso by alpha=0.01 is : ',lasso001.score(X_test,y_test))
# also the no.  of features included are also increased
print("The no. of features used by Laso by alpha=0.01 is: ",np.sum(lasso001.coef_!=0))

lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print("Training set score by lasso by alpha=0.0001 : {:.2f}".format(lasso00001.score(X_train, y_train))) 
print("Test set scoreby lasso by alpha=0.0001 : {:.2f}".format(lasso00001.score(X_test, y_test))) 
# Training set score by lasso by alpha=0.0001 : 0.95 and Test set scoreby lasso by alpha=0.0001 : 0.64
# Number of features used: 96 . BY decreasing alpha beyond a limit the features increased but it leads to overfitting and reduces regularization
print("Number of features used: {}".format(np.sum(lasso00001.coef_ != 0))) 

# We can compare Lasso of different alphas with ridge alpha=0.1 in a plot
plt.plot(lasso.coef_, 's', label="Lasso alpha=1") 
plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01") 
plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")
plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25) 
plt.xlabel("Coefficient index") 
plt.ylabel("Coefficient magnitude")
plt.show()
