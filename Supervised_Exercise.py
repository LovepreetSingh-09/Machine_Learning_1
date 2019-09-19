# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 06:14:25 2019

@author: user
"""

import graphviz
import mglearn
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
from IPython.display import display
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

from sklearn.datasets import load_digits
dig=load_digits()
print(dig.keys())
print('Shape of Multiclass Classifiction data : ',dig.data.shape)
print('Data For Classification :\n',dig.data,'\nTarget is :\n',dig.target,'\nTarget_names :\n',dig.target_names,'\nImages :\n',dig.images)
from sklearn.datasets import load_diabetes
diab=load_diabetes()
print(diab.keys())
print('Shape of Regression data : ',diab.data.shape)
print('Data For Regression :\n',diab.data,'\nTarget is :\n',diab.target,'\nFeature_names :\n',diab.feature_names)
X_trainc,X_testc,y_trainc,y_testc=train_test_split(dig.data,dig.target,random_state=0)
X_trainr,X_testr,y_trainr,y_testr=train_test_split(diab.data,diab.target,random_state=0)

knnc=KNeighborsClassifier(n_neighbors=3) # At n_neighbors=1 The score is maximum = 0.991
knnc.fit(X_trainc,y_trainc)
print('Training score :{:.3f}'.format(knnc.score(X_trainc,y_trainc)))# The Training score is :0.992
print('Test score :{:.3f}'.format(knnc.score(X_testc,y_testc))) # The Test score is :0.987

# KNeighbors Regressor does very bad in regression
knnr=KNeighborsRegressor(n_neighbors=30) # At n_neighbors=30 The score is maximum = 0.393
knnr.fit(X_trainr,y_trainr)
print('Training score :{:.3f}'.format(knnr.score(X_trainr,y_trainr))) # Training score :0.505
print('The Test score is :{:.3f}'.format(knnr.score(X_testr,y_testr))) # The Test score is :0.393

logre=LogisticRegression(C=0.01).fit(X_trainc,y_trainc) # At high regulariztion C=0.01, the gap between train and test score is minimum 
print('Training score :',logre.score(X_trainc,y_trainc))# Training score : 0.9836674090571641
print('Test score:',logre.score(X_testc,y_testc)) # Test score: 0.9622222

svm=LinearSVC(C=0.001).fit(X_trainc,y_trainc) # At high regulariztion C=0.001, the gap between train and test score is minimum 
print('Training score :',svm.score(X_trainc,y_trainc))# Training score : 0.9829250
print('Test score:',svm.score(X_testc,y_testc)) # Test score: 0.964444
# print('Slopes or coeficients : ',svm.coef_)
# print('Intercept : ',svm.intercept_)
X_new=X_trainc[:,:4]
# print(X_new)

lereg=LinearRegression().fit(X_trainr,y_trainr)
print('The Train_score or accuracy is : ',lereg.score(X_trainr,y_trainr))# The Train_score or accuracy is :  0.555437
print('The Test_score or accuracy is : ',lereg.score(X_testr,y_testr)) # The Test_score or accuracy is :  0.3594009

ridge=Ridge(alpha=0.1).fit(X_trainr,y_trainr)
print('The Train_score or accuracy is : ',ridge.score(X_trainr,y_trainr))# The Train_score or accuracy is : 0.55020
print('The Test_score or accuracy is : ',ridge.score(X_testr,y_testr)) # The Test_score or accuracy is :  0.36901969

lasso=Lasso(alpha=0.000001,max_iter=100000).fit(X_trainr,y_trainr)
print('The Train_score or accuracy by lasso is : ',lasso.score(X_trainr,y_trainr))# The Train_score or accuracy by lasso is :   0.555437
print('The Test_score or accuracy by lasso is : ',lasso.score(X_testr,y_testr))# The Test_score or accuracy by lasso is :   0.359400272
print('The features used by Lasso are : ',np.sum(lasso.coef_!=0))# The features used by Lasso are :  10

dec=DecisionTreeClassifier(max_depth=9,random_state=0).fit(X_trainc,y_trainc)
print('Training score :',dec.score(X_trainc,y_trainc))# Training score :  0.95471
print('Test score:',dec.score(X_testc,y_testc)) # Test score: 0.833333

der=DecisionTreeRegressor(max_depth=2,random_state=0).fit(X_trainr,y_trainr)
print('The Train_score or accuracy is : ',der.score(X_trainr,y_trainr))# The Train_score or accuracy is :   0.49639482
print('The Test_score or accuracy is : ',der.score(X_testr,y_testr)) # The Test_score or accuracy is :  0.105035

rfc=RandomForestClassifier(n_estimators=100).fit(X_trainc,y_trainc) # At high regulariztion C=0.01, the gap between train and test score is minimum 
print('Training score :',rfc.score(X_trainc,y_trainc))# Training score : 1.0
print('Test score:',rfc.score(X_testc,y_testc)) # Test score: 0.977777777

rfr=RandomForestRegressor(n_estimators=70).fit(X_trainr,y_trainr)
print('Training score :{:.3f}'.format(rfr.score(X_trainr,y_trainr)))# Training score :0.925
print('Test score :{:.3f}'.format(rfr.score(X_testr,y_testr))) # Test score :0.249

gbc=GradientBoostingClassifier(n_estimators=150).fit(X_trainc,y_trainc)
print('Training score :{:.3f}'.format(gbc.score(X_trainc,y_trainc)))# Training score : 1.00
print('Test score :{:.3f}'.format(gbc.score(X_testc,y_testc))) # Test score :0.964

gbr=GradientBoostingRegressor(n_estimators=20).fit(X_trainr,y_trainr)
print('Training score : {:.3f}'.format(gbr.score(X_trainr,y_trainr)))# Training score : 0.709
print('Test score : {:.3f}'.format(gbr.score(X_testr,y_testr)))# Test score :0.285


svc=SVC(kernel='rbf',C=10,gamma=0.1).fit(X_trainc,y_trainc)
print('Training score : {:.3f}'.format(svc.score(X_trainc,y_trainc)))# Training score : 1.00
print('Test score : {:.3f}'.format(svc.score(X_testc,y_testc))) # Test score : 0.084
# After Rescaling :-
X_trainc_scaled=scaler.fit_transform(X_trainc)
X_testc_scaled=scaler.fit_transform(X_testc)
svc=SVC(kernel='rbf',C=10,gamma=0.1).fit(X_trainc_scaled,y_trainc)
print('Training score : {:.3f}'.format(svc.score(X_trainc_scaled,y_trainc)))# Training score : 1.00
print('Test score : {:.3f}'.format(svc.score(X_testc_scaled,y_testc))) # Test score : 0.991

svr=SVR(kernel='rbf',C=15,gamma=10).fit(X_trainr,y_trainr)
print('Training score svr : {:.3f}'.format(svr.score(X_trainr,y_trainr))) # Training score svr : 0.566
print('Test score svr : {:.3f}'.format(svr.score(X_testr,y_testr))) # Test score svr : 0.389
# After Rescaling :-
scaler1=MinMaxScaler()
X_trainr_scaled=scaler.fit_transform(X_trainr)
X_testr_scaled=scaler.fit_transform(X_testr)
svr1=SVR(kernel='rbf',C=100,gamma=0.03).fit(X_trainr_scaled,y_trainr)
print('Training score : {:.3f}'.format(svr1.score(X_trainr_scaled,y_trainr))) # Training score : 0.499
print('Test score : {:.3f}'.format(svr1.score(X_testr_scaled,y_testr))) # Test score : 0.395

mlpc=MLPClassifier(max_iter=1000).fit(X_trainc,y_trainc)
print('Training score : {:.3f}'.format(mlpc.score(X_trainc,y_trainc)))# Training score : 1.00
print('Test score :{:.3f}'.format(mlpc.score(X_testc,y_testc))) # Test score :0.967
# After Rescaling :-
mlpc1=MLPClassifier(max_iter=1000,solver='lbfgs',random_state=0,hidden_layer_sizes=[10,10],activation='tanh').fit(X_trainc_scaled,y_trainc)
print('Training score : {:.3f}'.format(mlpc1.score(X_trainc_scaled,y_trainc)))# Training score : 1.00
print('Test score :{:.3f}'.format(mlpc1.score(X_testc_scaled,y_testc))) # wothout any argument in the MLP bracket Test score :0.978

mlpr=MLPRegressor(max_iter=10000).fit(X_trainr,y_trainr)
print('Training score : {:.3f}'.format(mlpr.score(X_trainr,y_trainr)))# Training score : 0.553
print('Test score :{:.3f}'.format(mlpr.score(X_testr,y_testr))) # Test score :0.354
# After Rescaling :-
mlpr1=MLPRegressor(max_iter=10000,solver='lbfgs',hidden_layer_sizes=[4,4],random_state=0).fit(X_trainr_scaled,y_trainr)
print('Training score : {:.3f}'.format(mlpr1.score(X_trainr_scaled,y_trainr)))# Training score : 0.555
print('Test score :{:.3f}'.format(mlpr1.score(X_testr_scaled,y_testr))) # Test score : 0.313

# For Classification, SVC(0.991), KNeighbors(0.987) and MLP(0.978) produced best score
# For Regression, SVR(0.395) ,KNeighbors(0.393) and Ridge(0.369) produced best score