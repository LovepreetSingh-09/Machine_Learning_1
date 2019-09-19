# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 21:28:42 2019

@author: user
"""

# Pipelines
# Pipeline class is a general-purpose tool to chain together multiple processing steps in a machine learning workflow.

import graphviz
import mglearn
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
from IPython.display import display
from sklearn.model_selection import train_test_split

# Here, by preprocessing (scaling), we get a good accuracy on default parameter of SVC
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer 
from sklearn.preprocessing import MinMaxScaler
# load and split the data 
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
# compute minimum and maximum on the training data
scaler = MinMaxScaler().fit(X_train)
 # rescale the training data
X_train_scaled = scaler.transform(X_train)
svm = SVC()
# learn an SVM on the scaled training data 
svm.fit(X_train_scaled, y_train)
# scale the test data and score the scaled data
X_test_scaled = scaler.transform(X_test)
print("Test score: {:.2f}".format(svm.score(X_test_scaled, y_test))) # 0.95

# Now we get better accuracy using GridSearchCV
from sklearn.model_selection import GridSearchCV
# for illustration purposes only, don't use this code! 
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100] ,'gamma': [0.001, 0.01, 0.1, 1, 10, 100]} 
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=5)
grid.fit(X_train_scaled, y_train)
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_)) # 0.98
print("Best parameters: ", grid.best_params_) # {'C': 1, 'gamma': 1}
print("Test set accuracy: {:.2f}".format(grid.score(X_test_scaled, y_test))) # 0.97
# Here we used all the scaled training data to fit in the cross-validation
# This means that the test split used in validation is also used for scaling which means model did'nt used on new data (test split).
# If we observe new data (test set), this data will not have been used to scale the training data and it might have a different minimum and maximum than the training data.
mglearn.plots.plot_improper_processing()
plt.show() 

# To get around this problem, the splitting of the dataset during cross-validation should be done before doing any preprocessing.
# To achieve this in scikit-learn with the cross_val_score function and the Grid SearchCV function, we can use the Pipeline class. 
# The Pipeline class itself has fit, predict, and score methods and behaves just like any other model in scikit-learn. 
# The most common use case of the Pipeline class is in chaining preprocessing steps (like scaling of the data) together with a supervised model like a classifier.
from sklearn.pipeline import Pipeline
pipe=Pipeline([('scaler',MinMaxScaler()),('svm',SVC())])
# scaler and svm are the names or instances given to their respective methods
# Here, pipe.fit first calls fit on the first step (the scaler), then transforms the training data using the scaler and finally fits the SVM with the scaled data. 
pipe.fit(X_train,y_train)
# Calling the score method on the pipeline first transforms the test data using the scaler and then calls the score method on the SVM using the scaled test data. 
print(pipe.score(X_test,y_test)) # 0.951
print(pipe.predict(X_test))

# Using Pipelines in Grid Searches
from sklearn.model_selection import GridSearchCV
param_grid = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100] ,'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
# We need to specify for each parameter which step of the pipeline it belongs to.
# This should be done by having double-underscore between the instance and parameter.
grid=GridSearchCV(pipe,param_grid=param_grid,cv=5)
grid.fit(X_train,y_train)
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_)) # 0.98
print("Test set score: {:.2f}".format(grid.score(X_test, y_test))) # 0.97
print("Best parameters: {}".format(grid.best_params_)) # {'svm__C': 1, 'svm__gamma': 1}
# Now for each split in the cross-validation, the MinMaxScaler is refit with only the training splits and no information is leaked from the test split into the parameter search. 
mglearn.plots.plot_proper_processing()
plt.show()

# Let’s consider a synthetic regression task with 100 samples and 10,000 features that are sampled independently from a Gaussian distribution. 
rnd = np.random.RandomState(seed=0) 
X = rnd.normal(size=(100, 10000)) 
y = rnd.normal(size=(100,))
# There is no relation between the data, X, and the target, y (they are independent), so it should not be possible to learn anything from this dataset. 
# First, select the most informative of the 10,000 features using SelectPercentile feature selection, and then we evaluate a Ridge regressor using cross-validation
from sklearn.feature_selection import SelectPercentile,f_regression
select=SelectPercentile(score_func=f_regression,percentile=5).fit(X,y)
X_selected = select.transform(X)
print("X_selected.shape: {}".format(X_selected.shape)) # (100, 500)
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
print(np.mean(cross_val_score(Ridge(), X_selected, y, cv=5))) # 0.91
# Because we fit the feature selection outside of the cross-validation, it could find features that are correlated both on the training and the test folds. 
# The information we leaked from the test folds was very informative, leading to highly unrealistic results.
# Let’s compare this to a proper cross-validation using a pipeline
pipe=Pipeline([('select',SelectPercentile(score_func=f_regression,percentile=5)),('ridge',Ridge())])
print(np.mean(cross_val_score(pipe,X,y,cv=5))) # -0.246
# Using the pipeline, the feature selection is now inside the cross-validation loop.
# This means features can only be selected using the training folds of the data, not the test fold.

# Working of fit mehod in pipeline :-
# Internally, during the call to Pipeline.fit, the pipeline calls fit and then transform on each step in turn,  with the input given by the output of the transform method of the previous step.
# For the last step in the pipeline, just fit is called.
# pipeline.steps is a list of tuples, so pipeline.steps[0][1] is the first estimator, pipe line.steps[1][1] is the second estimator, and so on
def fit(self, X, y):     
    X_transformed = X     
    for name, estimator in self.steps[:-1]:         
        # iterate over all but the final step
        # fit and transform the data
        X_transformed = estimator.fit_transform(X_transformed, y)
    # fit the last step
    self.steps[-1][1].fit(X_transformed, y)     
    return self
# When predicting using Pipeline, we similarly transform the data (test data) using all but the last step and then call predict on the last step.
def predict(self, X):     
    X_transformed = X     
    for step in self.steps[:-1]:
        # iterate over all but the final step
        # transform the data
        X_transformed = step[1].transform(X_transformed)
    # predict using the last step
    return self.steps[-1][1].predict(X_transformed)

# make_pipeline -
# There is a convenience function, make_pipeline, that will create a pipeline for us and automatically name each step based on its class. 
from sklearn.pipeline import make_pipeline
pipe_long = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC(C=100))])
pipe_short=make_pipeline(MinMaxScaler(),SVC())
# The pipeline objects pipe_long and pipe_short do exactly the same thing, but pipe_short has steps that were automatically named. 
print(pipe_short.steps)
# In general, the step names are just lowercase versions of the class names.
# But If multiple steps have the same class, a number is appended.
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
pipe = make_pipeline(StandardScaler(), PCA(n_components=2), StandardScaler()) 
print("Pipeline steps:\n{}".format(pipe.steps)) 
# First StandardScaler step was named standardscaler-1 and the second standardscaler-2. 

# Often you want to inspect attributes of one of the steps of the pipeline—say, the coefficients of a linear model or the components extracted by PCA.
# The easiest way to access the steps in a pipeline is via the named_steps attribute.
pipe.fit(cancer.data)
# # fit the pipeline defined before to the cancer dataset
print(pipe.named_steps['pca'].components_.shape) # (2, 30)

from sklearn.linear_model import LogisticRegression
pipe = make_pipeline(StandardScaler(), LogisticRegression())
param_grid = {'logisticregression__C': [0.01, 0.1, 1, 10, 100]}
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=4) 
grid = GridSearchCV(pipe, param_grid, cv=5) 
grid.fit(X_train, y_train)
# The best model found by GridSearchCV, trained on all the training data, is stored in grid.best_estimator_
print(grid.best_estimator_)
print(grid.best_estimator_.named_steps['logisticregression'])
print(grid.best_estimator_.named_steps['logisticregression'].coef_)

from sklearn.datasets import load_boston
boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)
from sklearn.preprocessing import PolynomialFeatures
pipe = make_pipeline(StandardScaler(),PolynomialFeatures(),Ridge())
param_grid = {'polynomialfeatures__degree': [1, 2, 3] ,'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1) 
grid.fit(X_train, y_train)
# 3 rows will be of polynomial degree and 6 columns of 6 parameter values
plt.matshow(grid.cv_results_['mean_test_score'].reshape(3,-1),vmin=0,cmap='viridis')
plt.xlabel("ridge__alpha") 
plt.ylabel("polynomialfeatures__degree")
plt.xticks(range(len(param_grid['ridge__alpha'])),param_grid['ridge__alpha'])
plt.yticks(range(len(param_grid['polynomialfeatures__degree'])) ,param_grid['polynomialfeatures__degree']) 
plt.colorbar()
print(grid.best_params_) # {'polynomialfeatures__degree': 2, 'ridge__alpha': 10}
print("Test-set score: {:.2f}".format(grid.score(X_test, y_test)))  # 0.77
# Now run the pipeline and gridsearch without polynomial feature
param_grid = {'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
pipe = make_pipeline(StandardScaler(), Ridge()) 
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
print("Score without poly features: {:.2f}".format(grid.score(X_test, y_test))) # 0.63
# hence, we come to know that using polynomial feature is actually help us to make a better model.

# Grid-Searching Which Model To Use :-
# Trying all possible solutions is usually not a viable machine learning strategy. 
# We know that the SVC might need the data to be scaled, so we also search over whether to use StandardScaler or no preprocessing.
# For the RandomForestClassifier, we know that no preprocessing is necessary.
# We want two steps, one for the preprocessing and then a classifier.
# We can instantiate this using SVC and StandardScaler.
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
# Wether we use RandomForest or SVC in classifier or Standerdscaler or None in preprocessing, it doesn't affect the model.
# We just need to initiate the instances of the methods to be perform.
pipe=Pipeline([('preprocessing',None),('classifier',RandomForestClassifier())])
# When we wanted to skip a step in the pipeline (for example, because we don’t need preprocessing for the RandomForest), we can set that step to None.
param_grid=[{'classifier':[SVC()],'preprocessing':[StandardScaler(),None],'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100] ,
                           'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]} ,{'classifier': [RandomForestClassifier(n_estimators=100)],
                                            'preprocessing': [None], 'classifier__max_features': [1, 2, 3]}]
grid=GridSearchCV(pipe,param_grid=param_grid,cv=5)
grid.fit(X_train,y_train)
print(grid.best_params_)
print(grid.best_score_) # 0.985
print(grid.score(X_test,y_test)) # 0.979






