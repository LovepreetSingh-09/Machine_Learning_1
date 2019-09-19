# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 17:46:07 2019

@author: user
"""

# Model Evaluation and Improvement 

import graphviz
import mglearn
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
from IPython.display import display
from sklearn.model_selection import train_test_split

# Cross Validation :-
# k-fold cross-validation, where k is a user-specified number, usually 5 or 10.
# Each Split consist of total data samples of Dataset.
# When performing five-fold cross-validation, the data is first partitioned into five parts of (approximately) equal size, called folds.
mglearn.plots.plot_cross_validation()
plt.show() 
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris 
from sklearn.linear_model import LogisticRegression
iris=load_iris()
logreg=LogisticRegression()
scores=cross_val_score(logreg,iris.data,iris.target,cv=5) # For Integer/None input to cv, if y is binary or multiclass classifier stratified Cross Validation is used.
print(scores) # [1.  0.96666667   0.93333333   0.9    1. ]
print(scores.mean()) # 0.96000
# Cross Validation mostly used for Regression while stratified Cross Validation is used for the Classification.

# Stratified Cross Validation :-
# In stratified cross-validation, we split the data such that the proportions or percentage between classes are the same in each fold as they are in the whole dataset
mglearn.plots.plot_stratified_cross_validation()
plt.show()
from sklearn.model_selection import KFold
kfold=KFold(n_splits=5)
scores=cross_val_score(logreg,iris.data,iris.target,cv=kfold)
print(scores) # [1.  0.93333333  0.43333333  0.96666667  0.43333333]
print(scores.mean()) # 0.7533
kfold=KFold(n_splits=3)
scores=cross_val_score(logreg,iris.data,iris.target,cv=kfold)
print(scores) # [0.  0.  0.]
print(scores.mean()) # 0.0
kfold=KFold(n_splits=3,shuffle=True,random_state=0)
scores=cross_val_score(logreg,iris.data,iris.target,cv=kfold)
print(scores) # [0.9  0.96 0.96]
print(scores.mean()) # 0.94

# Leave-one-out cross-validation :-
# leave-one-out cross-validation is like a k-fold cross-validation where each fold is a single sample. 
# For each split, you pick a single data point to be the test set. 
# This can be very time consuming, particularly for large datasets, but sometimes provides better estimates on small datasets
from sklearn.model_selection import LeaveOneOut
loo=LeaveOneOut()
scores=cross_val_score(logreg,iris.data,iris.target,cv=loo)
print(len(scores)) # 150
print(scores.mean()) # 0.95333
# Here it is like 150 X 150 array. 150 folds has whole dataset (150 samples) and the test data sample changes with every fold from 1 to 150.

# Shuffle-split cross-validation :-
# Each split samples train_size many points for the training set and test_size many (disjoint) point for the test set.
# This splitting is repeated n_splits times.  
mglearn.plots.plot_shuffle_split() # ShuffleSplit with 10 points, train_size=5, test_size=2, and n_splits=4
plt.show()
from sklearn.model_selection import ShuffleSplit
ss=ShuffleSplit(train_size=.7,test_size=.2,n_splits=7)
scores=cross_val_score(logreg,iris.data,iris.target,cv=ss)
print(scores) # [0.83333333 0.83333333 0.83333333 0.93333333 0.86666667 0.93333333 0.93333333]
print(scores.mean()) # 0.880

# Cross-validation with groups :-
# GroupKFold takes an array of groups as argument.
# The groups array here indicates groups in the data that should not be split when creating the training and test sets, and should not be confused with the class label.
from sklearn.model_selection import GroupKFold
gr=GroupKFold()
# assume the first three samples belong to the same group,
# then the next four, etc. 
groups=[0,0,0,1,1,1,1,2,2,3,3,3]
X,y=mglearn.datasets.make_blobs(n_samples=12,random_state=0)
scores=cross_val_score(logreg,X,y,groups,cv=gr)
print(scores) # [1.  0.8   1. ]
print(scores.mean())  #  0.9333
mglearn.plots.plot_group_kfold()


# Grid Search :-
# To find the best parameters for a model.
# naive grid search implementation 
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
print("Size of training set: {}   size of test set: {}".format( X_train.shape[0], X_test.shape[0])) 
# 112   38
best_score=0
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:     
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        # for each combination of parameters, train an SVC
        svm = SVC(gamma=gamma, C=C)         
        svm.fit(X_train, y_train)
        # evaluate the SVC on the test set         
        score = svm.score(X_test, y_test)         # if we got a better score, store the score and parameters        
        if score > best_score:             
            best_score = score
            best_parameters={'C':C,'gamma':gamma}
print("Best score: {:.2f}".format(best_score))  #  0.97
print("Best parameters: {}".format(best_parameters)) # {'C': 100, 'gamma': 0.001}
# We tried many different parameters and selected the one with best accuracy on the test set, but this accuracy won’t necessarily carry over to new data.
# Because we used the test data to adjust the parameters, we can no longer use it to assess how good the model is. 
# One way to resolve this problem is to split the data again, so we have three sets:
# the training set to build the model, the validation (or development) set to select the parameters of the model, and the test set to evaluate the performance of the selected parameters.
mglearn.plots.plot_threefold_split()
plt.show()
X_trainval,X_test,y_trainval,y_test=train_test_split(iris.data,iris.target,random_state=0)
X_train,X_valid,y_train,y_valid=train_test_split(X_trainval,y_trainval,random_state=1)
print("Size of training set: {}   size of validation set: {}   size of test set: {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0])) 
# 84   28   38
best_score = 0
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:     
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm=SVC(gamma=gamma,C=C)
        svm.fit(X_train,y_train)
        score=svm.score(X_valid,y_valid)
         # if we got a better score, store the score and parameters        
        if score > best_score:             
             best_score = score
             best_parameters = {'C': C, 'gamma': gamma}
# rebuild a model on the combined training and validation set,
# and evaluate it on the test set 
svm=SVC(**best_parameters)
svm.fit(X_trainval,y_trainval)
test_score=svm.score(X_test,y_test)
print("Best score on validation set: {:.2f}".format(best_score))  #  0.96
print("Best parameters: ", best_parameters) #  {'C': 10, 'gamma': 0.001}
print("Test set score with best parameters: {:.2f}".format(test_score)) # 0.92

# Grid Search with Cross Validation :-
best_score=0
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:     
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:         
        # for each combination of parameters,
        # train an SVC         
        svm = SVC(gamma=gamma, C=C)        
        # perform cross-validation
        scores = cross_val_score(svm, X_trainval, y_trainval, cv=5)
        # compute mean cross-validation accuracy
        score = np.mean(scores)         
        # if we got a better score, store the score and parameters         
        if score > best_score:             
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}
# Total 36 * 5 = 180 models were built
# rebuild a model on the combined training and validation set
svm = SVC(**best_parameters) 
svm.fit(X_trainval, y_trainval)
test_score=svm.score(X_test,y_test)
print("Best score on validation set: {:.2f}".format(best_score))  #  0.97
print("Best parameters: ", best_parameters) #  {'C': 100, 'gamma': 0.01}
print("Test set score with best parameters: {:.2f}".format(test_score)) # 0.97
mglearn.plots.plot_cross_val_selection()
plt.show()
mglearn.plots.plot_grid_search_overview()
plt.show()
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100] ,'gamma': [0.001, 0.01, 0.1, 1, 10, 100]} 
print("Parameter grid:\n{}".format(param_grid))
from sklearn.model_selection import GridSearchCV
gridsearch=GridSearchCV(SVC(),param_grid,cv=5)
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
gridsearch.fit(X_train,y_train)
score=gridsearch.score(X_test,y_test)
print(score)  # 0.9736
print(gridsearch.best_params_) # {'C': 100, 'gamma': 0.01}
# For best cross-validation accuracy (the mean accuracy over the different splits for this parameter setting) 
print(gridsearch.best_score_)  # 0.97321
# the best parameters trained on the whole training set 
print(gridsearch.best_estimator_)
# SVC(C=100, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
print(gridsearch.cv_results_)
results=pd.DataFrame(gridsearch.cv_results_)
print(results.head())
scores=np.array(results.mean_test_score).reshape(6,6)
mglearn.tools.heatmap(scores,xlabel='gamma',xticklabels=param_grid['gamma'],ylabel='C',yticklabels=param_grid['C'],cmap='viridis')
plt.show()
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
param_grid_linear = {'C': np.linspace(1, 2, 6) , 'gamma':  np.linspace(1, 2, 6)}
param_grid_one_log = {'C': np.linspace(1, 2, 6) ,'gamma':  np.logspace(-3, 2, 6)}
param_grid_range = {'C': np.logspace(-3, 2, 6) , 'gamma':  np.logspace(-7, -2, 6)}
for param_grid, ax in zip([param_grid_linear, param_grid_one_log, param_grid_range], axes):
     grid_search = GridSearchCV(SVC(), param_grid, cv=5)
     grid_search.fit(X_train, y_train)
     scores = grid_search.cv_results_['mean_test_score'].reshape(6, 6)
    # plot the mean cross-validation scores     
     scores_image = mglearn.tools.heatmap(
        scores, xlabel='gamma', ylabel='C', xticklabels=param_grid['gamma'] , yticklabels=param_grid['C'], cmap="viridis", ax=ax)
plt.colorbar(scores_image, ax=axes)
plt.show()
# Make multiple dictionaries in a list
param_grid = [{'kernel': ['rbf'] ,'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]},{'kernel': ['linear'] ,'C': [0.001, 0.01, 0.1, 1, 10, 100]}] 
print("List of grids:\n{}".format(param_grid))
from sklearn.model_selection import GridSearchCV
gridsearch=GridSearchCV(SVC(),param_grid,cv=5)
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
gridsearch.fit(X_train,y_train)
score=gridsearch.score(X_test,y_test)
print(gridsearch.best_params_) # {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}
print(gridsearch.best_score_) # 0.9732
results=pd.DataFrame(gridsearch.cv_results_)
display(results)

# Nested Cross-Validation with Grid Search :-
from sklearn.model_selection import StratifiedKFold
for t,ts in StratifiedKFold(5).split(iris.data,iris.target):
    print(t.shape,ts.shape)
    for th,tsh in StratifiedKFold(5).split(t,iris.target[t]):
        print(th.shape,tsh.shape)
# here first split will occur (120, 30) ,then five splits of (96, 24) of that 120 will happen.
# The grid search will occur on that splits checking 36 models on each five splits.
# Then the best parameter found will be use to build the model of the outer split and score will be measured.
# Then again next five splits of outer split will occur. Thus, the process goes on.
# Hence,5 * 36 * 5 = 900 models will be built
scores=cross_val_score(GridSearchCV(SVC(),param_grid,cv=5),iris.data,iris.target,cv=5)
# Score is the score of outer splits
print("Cross-validation scores: ", scores) # [0.96667  1.  0.9  0.96666667  1. ]
print("Mean cross-validation score: ", scores.mean())  # 0.966
 # Representation :-
def nested_cv(X, y, inner_cv, outer_cv, Classifier, parameter_grid):     
    outer_scores = []
    # for each split of the data in the outer cross-validation
    # (split method returns indices of training and test parts)     
    for training_samples, test_samples in outer_cv.split(X, y):
        # find best parameter using inner cross-validation
        best_params = {}         
        best_score = -np.inf         
        # iterate over parameters         
        for parameters in parameter_grid:
            # accumulate score over inner splits
            cv_scores = []
            # iterate over inner cross-validation             
            for inner_train, inner_test in inner_cv.split(X[training_samples], y[training_samples]):
                # build classifier given parameters and training data
                clf = Classifier(**parameters)                 
                clf.fit(X[inner_train], y[inner_train])
                # evaluate on inner test set
                score = clf.score(X[inner_test], y[inner_test])
                cv_scores.append(score)
            # compute mean score over inner folds             
            mean_score = np.mean(cv_scores)             
            if mean_score > best_score:
                # if better than so far, remember parameters
                best_score = mean_score                 
                best_params = parameters
        # build classifier on best parameters using outer training set
        clf = Classifier(**best_params)
        clf.fit(X[training_samples], y[training_samples])
        # evaluate
        outer_scores.append(clf.score(X[test_samples], y[test_samples]))     
    return np.array(outer_scores)
from sklearn.model_selection import ParameterGrid, StratifiedKFold 
print(param_grid)
scores = nested_cv(iris.data, iris.target, StratifiedKFold(5) ,StratifiedKFold(5), SVC, ParameterGrid(param_grid)) 
print("Cross-validation scores: {}".format(scores)) #   [0.96666667   1.   0.96666667   0.96666667   1. ]
# While running a grid search over many parameters and on large datasets can be computationally challenging, it is also embarrassingly parallel. 


# Metrics :-
# 1.) Binary Classification :-
# Often, accuracy is not a good measure of predictive performance, as the number of mistakes we make does not contain all the information we are interested in. 
# For any application, we need to ask ourselves what the consequences of these mistakes might be in the real world.
# An incorrect positive prediction is called a false positive. 
# An incorrect negative prediction—is called a false negative. 
# In statistics, a false positive is also known as type I error, and a false negative as type II error. 
# Datasets in which one class is much more frequent than the other are often called imbalanced datasets, or datasets with imbalanced classes. 
# To illustrate, we’ll create a 9:1 imbalanced dataset from the digits dataset, by classifying the digit 9 against the nine other classes
from sklearn.datasets import load_digits
digits = load_digits() 
y = digits.target == 9
print(y)
X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=0)
from sklearn.dummy import DummyClassifier
dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
pred_most_frequent = dummy_majority.predict(X_test)
print("Unique predicted labels: {}".format(np.unique(pred_most_frequent))) #  [False]
print("Test score: {:.2f}".format(dummy_majority.score(X_test, y_test)))  # 0.90
# We obtained close to 90% accuracy without learning anything. 
# It is achieved by predicting just one class. So, this model is of no use.
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
pred_tree = tree.predict(X_test)
print("Test score: {:.2f}".format(tree.score(X_test, y_test))) # 0.92
# According to accuracy, the DecisionTreeClassifier is only slightly better than the constant predictor.
# This indicate that accuracy is in fact not a good measure here.
# let’s evaluate two more classifiers, LogisticRegression and the default DummyClassifier, which makes random predictions but produces classes with the same proportions as in the training set
from sklearn.linear_model import LogisticRegression
dummy = DummyClassifier().fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)
print("dummy score: {:.2f}".format(dummy.score(X_test, y_test))) # 0.81
logreg = LogisticRegression(C=0.1).fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)
print("logreg score: {:.2f}".format(logreg.score(X_test, y_test))) # 0.98
# The problem here is that accuracy is an inadequate measure for quantifying predictive performance in this imbalanced setting. 
# In particular, we would like to have metrics that tell us how much better a model is than making “most frequent” predictions or random predictions, as they are computed in pred_most_frequent and pred_dummy. 
# Confusion Matrix :-
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,pred_logreg)
print("Confusion matrix:\n{}".format(confusion))
# [[401   2]
# [  8  39]] 
# The output of confusion_matrix is a 2 X 2 array, where the rows correspond to the true classes and the columns correspond to the predicted classes.
# Here, 401 is True Negative ; 39 is TP ; 8 is FN ; 2 is FP.
mglearn.plots.plot_confusion_matrix_illustration()
plt.show()
mglearn.plots.plot_binary_confusion_matrix()
plt.show()
print("Most frequent class:")
print(confusion_matrix(y_test, pred_most_frequent))
print("\nDummy model:")
print(confusion_matrix(y_test, pred_dummy))
print("\nDecision tree:")
print(confusion_matrix(y_test, pred_tree))
print("\nLogistic Regression") 
print(confusion_matrix(y_test, pred_logreg))
# Here, logreg did best while dummy matrix with strategy 'most_frequent' performed worst
# Logreg performed better than Decision Tree
# We always want the first diagonal to be as high as possible because this diagonal is TN and TP.
# And the second diagonal to be as low s possible which is FP and FN.

# Accuracy = = (TP+TN) / (TP+TN+FP+FN)

# Precision :-
# Precision measures how many of the samples predicted as positive are actually positive
# TP / (TP + FP)
# Precision is used as a performance metric when the goal is to limit the number of false positives. 
# Precision is also known as positive predictive value (PPV).

# Recall :-
# Recall measures how many of the positive samples are captured by the positive predictions
# TP / (TP + FN)
# Recall is used as performance metric when we need to identify all positive samples.
# Other names for recall are sensitivity, hit rate, or true positive rate (TPR).
# You can trivially obtain a perfect recall if you predict all samples to belong to the positive class— there will be no false negatives, and no true negatives either.
# However, predicting all samples as positive will result in many false positives, and therefore the precision will be very low.
# On the other hand, if you find a model that predicts only the single data point it is most sure about as positive and the rest as negative, then precision will be perfect (assuming this data point is in fact positive), but recall will be very bad.

# So, whilZ precision and recall are very important measures, looking at only one of them will not provide you with the full picture. One way to summarize them is the f-score or f-measure, which is with the harmonic mean of precision and recall

# f1_score :-
# f1_score = (2 * Precision * Recall) / (Precision + Recall)
# it takes precision and recall into account, it can be a better measure than accuracy on imbalanced binary classification datasets.
from sklearn.metrics import f1_score 
print("f1 score most frequent: {:.2f}".format(f1_score(y_test, pred_most_frequent))) # 0.0
print("f1 score dummy: {:.2f}".format(f1_score(y_test, pred_dummy))) # 0.07
print("f1 score tree: {:.2f}".format(f1_score(y_test, pred_tree))) # 0.55
print("f1 score logistic regression: {:.2f}".format(f1_score(y_test, pred_logreg))) # 0.89

# classification_report :-
from sklearn.metrics import classification_report 
print(classification_report(y_test, pred_most_frequent,target_names=["not nine", "nine"]))
print(classification_report(y_test, pred_dummy,target_names=["not nine", "nine"])) 
print(classification_report(y_test, pred_logreg, target_names=["not nine", "nine"])) 

# Taking uncertainty into account :-
# The following is an example of an imbalanced binary classification task, with 400 points in the negative class classified against 50 points in the positive class. 
from mglearn.datasets import make_blobs
# X is (450, 2)
X, y = make_blobs(n_samples=(400, 50), centers=2, cluster_std=[7.0, 2] ,random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) 
svc = SVC(gamma=.05).fit(X_train, y_train) 
mglearn.plots.plot_decision_threshold()
print(classification_report(y_test, svc.predict(X_test))) 
# Default threshold for decision function is 0
# More threshold means lesser area for the prediction and vice-versa
y_pred_lower_threshold = svc.decision_function(X_test) > -.8 
print(y_pred_lower_threshold)
print(classification_report(y_test, y_pred_lower_threshold)) 
print(X.shape)

# Calibration:  a calibrated model is a model that provides an accurate measure of its uncertainty.

# Precision-recall Curve :-
# To look at all possible thresholds, or all possible trade-offs of precision and recalls at once. 
# This is possible using a tool called the precision-recall curve. 
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, svc.decision_function(X_test))
print(precision)
print(recall,thresholds)

# Use more data points for a smoother curve
X, y = make_blobs(n_samples=(4000, 500), centers=2, cluster_std=[7.0, 2] ,random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
svc = SVC(gamma=.05).fit(X_train, y_train) 
precision, recall, thresholds = precision_recall_curve(y_test, svc.decision_function(X_test))
# find threshold closest to zero 
close_zero = np.argmin(np.abs(thresholds))
print(thresholds.shape) # (1090,)
print(close_zero) # 964
plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10,label="threshold zero", fillstyle="none", c='k', mew=2)
plt.plot(precision, recall, label="precision recall curve")
plt.xlabel("Precision") 
plt.ylabel("Recall") 
plt.legend(loc="best")
plt.show()
# Precision-Recall Curve using Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=0, max_features=2) 
rf.fit(X_train, y_train)
# RandomForestClassifier has predict_proba, but not decision_function 
precision_rf, recall_rf, thresholds_rf = precision_recall_curve(y_test, rf.predict_proba(X_test)[:, 1]) 
plt.plot(precision, recall, label="svc")
plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10,label="threshold zero svc", fillstyle="none", c='k', mew=2) 
plt.plot(precision_rf, recall_rf, label="rf")
print(rf.predict_proba(X_test)[:,1])
close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))
print(close_default_rf ) # 47
plt.plot(precision_rf[close_default_rf], recall_rf[close_default_rf], '^', c='k',markersize=10, label="threshold 0.5 rf", fillstyle="none", mew=2)
plt.xlabel("Precision") 
plt.ylabel("Recall") 
plt.legend(loc="best")
plt.show()
# Here, random forest performs better at the extremes, for very high recall or very high precision requirements.
# Around the middle (approximately precision=0.7), the SVM performs better.
# The f1-score only captures one point on the precision-recall curve, the one given by the default threshold.
print("f1_score of random forest: {:.3f}".format(f1_score(y_test, rf.predict(X_test)))) # 0.610
print("f1_score of svc: {:.3f}".format(f1_score(y_test, svc.predict(X_test))))  # 0.656

# average_precision_score :-
# One particular way to summarize the precision-recall curve is by computing the integral or area under the curve of the precision-recall curve, also known as the average precision.
# You can use the average_precision_score function to compute the average precision.
# Because we need to compute the precision-recall curve and consider multiple thresholds, the result of decision_function or pre dict_proba needs to be passed to average_precision_score, not the result of predict.
from sklearn.metrics import average_precision_score
ap_rf = average_precision_score(y_test, rf.predict_proba(X_test)[:, 1]) 
ap_svc = average_precision_score(y_test, svc.decision_function(X_test)) 
print("Average precision of random forest: {:.3f}".format(ap_rf)) # 0.660
print("Average precision of svc: {:.3f}".format(ap_svc))  # 0.666
# Both models perform similarly well.
# Average precision is the area under a curve that goes from 0 to 1, average precision always returns a value between 0 (worst) and 1 (best). 

# Receiver operating characteristics (ROC) and AUC :-
# To analyze the behavior of classifiers at different thresholds.
# It shows the plot of false positive rate (FPR) against the true positive rate (TPR). 
# Recall is also known as TPR.
# FPR is the fraction of false positives out of all negative samples
# FPR = FP / (FP + TN)
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, svc.decision_function(X_test))
plt.plot(fpr, tpr, label="ROC Curve")
plt.xlabel("FPR") 
plt.ylabel("TPR (recall)")
# find threshold closest to zero 
close_zero = np.argmin(np.abs(thresholds))
plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10,label="threshold zero", fillstyle="none", c='k', mew=2) 
plt.legend(loc=4)
plt.show()
# For the ROC curve, the ideal curve is close to the top left: you want a classifier that produces a high recall while keeping a low false positive rate. 
# roc_curve for random forest
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label="ROC Curve SVC") 
plt.plot(fpr_rf, tpr_rf, label="ROC Curve RF")
plt.xlabel("FPR") 
plt.ylabel("TPR (recall)")
plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10,label="threshold zero SVC", fillstyle="none", c='k', mew=2)
close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))
plt.plot(fpr_rf[close_default_rf], tpr[close_default_rf], '^', markersize=10,label="threshold 0.5 RF", fillstyle="none", c='k', mew=2) 
plt.legend(loc=4)
plt.show()

# AUC = Area under Curve
# The AUC can be interpreted as evaluating the ranking of positive samples.
# For classification problems with imbalanced classes, using AUC for model selection is often much more meaningful than using accuracy.
# Predicting randomly always produces an AUC of 0.5, no matter how imbalanced the classes in a dataset are.
from sklearn.metrics import roc_auc_score
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
svc_auc = roc_auc_score(y_test, svc.decision_function(X_test)) 
print("AUC for Random Forest: {:.3f}".format(rf_auc)) # 0.937
print("AUC for SVC: {:.3f}".format(svc_auc)) # 0.916
# Here, random forest performs better.
# Go back to the digits data for 9 prediction
y = digits.target == 9
X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=0) 
plt.figure()
for gamma in [1, 0.1, 0.01]:     
    svc = SVC(gamma=gamma).fit(X_train, y_train)     
    accuracy = svc.score(X_test, y_test)
    auc = roc_auc_score(y_test, svc.decision_function(X_test))     
    fpr, tpr, _ = roc_curve(y_test , svc.decision_function(X_test))     
    print("gamma = {:.2f}  accuracy = {:.2f}  AUC = {:.2f}".format(gamma, accuracy, auc))
    plt.plot(fpr, tpr, label="gamma={:.3f}".format(gamma))
plt.xlabel("FPR") 
plt.ylabel("TPR") 
plt.xlim(-0.01, 1) 
plt.show()
# The accuracy of all three settings of gamma is the same, 90%. 
# With gamma=0.01, we get a perfect AUC of 1.0. 
# For this reason, we highly recommend using AUC when evaluating models on imbalanced data. 

# Metrics for Multiclass Classification :-
# Basically, all metrics for multiclass classification are derived from binary classification metrics, but averaged over all classes.
# Accuracy is not a great evaluation measure here.
# Common tools are the confusion matrix and the classification report we saw in the binary case in the previous section.  
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split( digits.data, digits.target, random_state=0) 
lr = LogisticRegression().fit(X_train, y_train)
pred = lr.predict(X_test)
print("Accuracy: {:.3f}".format(accuracy_score(y_test, pred))) # 0.953
print("Confusion matrix:\n{}".format(confusion_matrix(y_test, pred))) 
# In this also, rows represent True Value while columns represent Prediction
print(classification_report(y_test,pred))
scores_image = mglearn.tools.heatmap(confusion_matrix(y_test, pred), xlabel='Predicted label',ylabel='True label', xticklabels=digits.target_names, yticklabels=digits.target_names, cmap=plt.cm.gray_r, fmt="%d")
plt.title("Confusion matrix") 
plt.gca().invert_yaxis()
plt.show()
# f1_score has 3 types :-
# 1.) "macro" averaging computes the unweighted per-class f-scoresor mean of each class. This gives equal weight to all classes, no matter what their size is.
# 2.) "weighted" averaging computes the mean of the per-class f- scores, weighted by their support. This is what is reported in the classification report.
# 3.) "micro" averaging computes the total number of false positives, false negatives, and true positives over all classes, and then computes precision, recall, and fscore using these counts.
# If you care about each sample equally much, it is recommended to use the "micro" average f1-score.
# And if you care about each class equally much, it is recommended to use the "macro" average f1- score.
print("Micro average f1 score: {:.3f}".format(f1_score(y_test, pred, average="micro"))) # 0.953
print("Macro average f1 score: {:.3f}".format(f1_score(y_test, pred, average="macro"))) # 0.954

# Using Evaluation Metrics in Model Selection :-
# 1.) Cross-Validation :-
# default scoring for classification is accuracy
print("Default scoring: {}".format(cross_val_score(SVC(), digits.data, digits.target == 9))) # [0.89983306 0.89983306 0.89983306]
# providing scoring="accuracy" doesn't change the results
explicit_accuracy =  cross_val_score(SVC(), digits.data, digits.target == 9,scoring="accuracy")
print("Explicit accuracy scoring: {}".format(explicit_accuracy))  # [0.89983306 0.89983306 0.89983306]
roc_auc =  cross_val_score(SVC(), digits.data, digits.target == 9, scoring="roc_auc")
print("AUC scoring: {}".format(roc_auc))  # [0.99372294 0.98957947 0.99594929]

# 2.) GridSearchCV with Accuracy :-
# The model of a certain parameter will make on the basis of accuracy score.
X_train, X_test, y_train, y_test = train_test_split(     digits.data, digits.target == 9, random_state=0)
# we provide a somewhat bad grid to illustrate the point:
param_grid = {'gamma': [0.0001, 0.01, 0.1, 1, 10]}
# using the default scoring of accuracy:
grid = GridSearchCV(SVC(), param_grid=param_grid)
grid.fit(X_train, y_train) 
print("Grid-Search with accuracy") 
print("Best parameters:", grid.best_params_) # {'gamma': 0.0001}
print("Best cross-validation score (accuracy)): {:.3f}".format(grid.best_score_))  # 0.970
print("Test set AUC: {:.3f}".format(roc_auc_score(y_test, grid.decision_function(X_test)))) # 0.992
print("Test set accuracy: {:.3f}".format(grid.score(X_test, y_test))) # 0.973

# # 2.) GridSearchCV with roc_auc_score :-
# The model of a certain parameter will make on the basis of roc_auc_score.
# using AUC scoring instead: 
grid = GridSearchCV(SVC(), param_grid=param_grid, scoring="roc_auc")
grid.fit(X_train, y_train) 
print("\nGrid-Search with AUC") 
print("Best parameters:", grid.best_params_) # {'gamma': 0.01}
print("Best cross-validation score (AUC): {:.3f}".format(grid.best_score_)) # 0.997
print("Test set AUC: {:.3f}".format(roc_auc_score(y_test, grid.decision_function(X_test)))) # 1.00
print("Test set accuracy: {:.3f}".format(grid.score(X_test, y_test))) # 1.00
# Hence, the best parameter changes on the basis of scoring method.

# Full list of supported arguments :-
from sklearn.metrics.scorer import SCORERS
print("Available scorers:\n{}".format(sorted(SCORERS.keys()))) 


