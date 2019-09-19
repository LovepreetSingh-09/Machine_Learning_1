# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:40:29 2019

@author: user
"""

# Automatic Feature Selection
# It can be a good idea to reduce the number of features to only the most useful ones and discard the rest. 
# This can lead to simpler models that generalize better. 
# There are three basic strategies: univariate statistics, model-based selection, and iterative selection. 
# . All of these methods are supervised methods, meaning they need the target for fitting the model. 
# This means we need to split the data into training and test sets, and fit the feature selection only on the training part of the data. 

import graphviz
import mglearn
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
from IPython.display import display
from sklearn.model_selection import train_test_split

# 1.) Univariate Statistics :-
# In this, we compute whether there is a statistically significant relationship between each feature and the target.
# Then the features that are related with the highest confidence are selected. In the case of classification, this is also known as analysis of variance (ANOVA).
# A key property of these tests is that they are univariate, meaning that they only consider each feature individually.
# Consequently, a feature will be discarded if it is only informative when combined with another feature.
# Univariate tests are often very fast to compute, and don’t require building a model. On the other hand, they are completely independent of the model that you might want to apply after the feature selection.
# The methods differ in how they compute the threshold, with the simplest ones being SelectKBest, which selects a fixed number k of features and SelectPercentile which selects a fixed percentage of features. 
# To use univariate feature selection in scikit-learn, you need to choose a test, usually either f_classif (the default) for classification or f_regression for regression.
from sklearn.datasets import load_breast_cancer 
from sklearn.feature_selection import SelectPercentile 
cancer = load_breast_cancer()
# get deterministic random numbers 
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data),50))
# add noise features to the data
# the first 30 features are from the dataset, the next 50 are noise
X_w_noise = np.hstack([cancer.data, noise])
X_train, X_test, y_train, y_test = train_test_split(X_w_noise, cancer.target, random_state=0, test_size=.5)
# use f_classif (the default) and SelectPercentile to select 50% of features
select=SelectPercentile(percentile=50)
select.fit(X_train,y_train)
X_train_selected=select.transform(X_train)
print("X_train.shape: {}".format(X_train.shape)) # (284, 80)
print("X_train_selected.shape: {}".format(X_train_selected.shape)) # (284, 40)
# We can find out which features have been selected using the get_support method, which returns a Boolean mask of the selected features 
mask = select.get_support()
print(mask)
# visualize the mask -- black is True, white is False 
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index") 
plt.yticks(()) 
plt.show()
# As you can see, most of the selected features are the original features and most of the noise features were removed.
# However, the recovery of the original features is not perfect. 
from sklearn.linear_model import LogisticRegression
# transform test data
X_test_selected = select.transform(X_test)
lr = LogisticRegression()
lr.fit(X_train, y_train)
print("Score with all features: {:.3f}".format(lr.score(X_test, y_test))) # 0.930
lr.fit(X_train_selected, y_train)
print("Score with only selected features: {:.3f}".format(lr.score(X_test_selected, y_test))) # 0.940

# Model-Based Feature Selection :- (SelectFromModel)
# Model-based feature selection uses a supervised machine learning model to judge the importance of each feature and keeps only the most important ones.
# The supervised model that is used for feature selection doesn’t need to be the same model that is used for the final supervised modeling.
# The feature selection model needs to provide some measure of importance for each feature, so that they can be ranked by this measure.
# Decision trees and decision tree–based models provide a feature_importances_ attribute, which directly encodes the importance of each feature.
# Linear models have coefficients, which can also be used to capture feature importances by considering the absolute values. 
# In contrast to univariate selection, model-based selection considers all features at once, and so can capture interactions (if the model can capture them). 
from sklearn.feature_selection import SelectFromModel 
from sklearn.ensemble import RandomForestClassifier
select=SelectFromModel(RandomForestClassifier(n_estimators=100,random_state=42),threshold='median')
# The SelectFromModel class selects all features that have an importance measure of the feature greater than the provided threshold.
# So, alll the features having importance lesser than the median will be discarded.
select.fit(X_train,y_train)
X_train_l1=select.transform(X_train)
print("X_train.shape: {}".format(X_train.shape)) # (284, 80)
print("X_train_l1.shape: {}".format(X_train_l1.shape)) # (284, 40)
mask = select.get_support()
# visualize the mask -- black is True, white is False 
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")
plt.yticks(())
plt.show()
# This is a quite complex model and much more powerful than using univariate tests. 
X_test_l1 = select.transform(X_test)
score = LogisticRegression().fit(X_train_l1, y_train).score(X_test_l1, y_test) 
print("Test score: {:.3f}".format(score)) # 0.951
# With the better feature selection, we also gained some improvements here.

# Iterative Feature Selection :-
# In iterative feature selection, a series of models are built, with varying numbers of features.
# There are two basic methods: 
# 1.) Starting with no features and adding features one by one until some stopping criterion is reached. 
# 2.) Starting with all features and removing features one by one until some stopping criterion is reached.
# Because a series of models are built, these methods are much more computationally expensive than the methods we discussed previously.
# One particular method of this kind is recursive feature elimination (RFE), which starts with all features, builds a model and discards the least important feature according to the model. 
from sklearn.feature_selection import RFE
select=RFE(RandomForestClassifier(n_estimators=100,random_state=42),n_features_to_select=40)
select.fit(X_train,y_train)
X_train_rfe=select.transform(X_train)
X_test_rfe=select.transform(X_test)
# visualize the selected features:
mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")
plt.yticks(())
plt.show()
# The feature selection got better compared to the univariate and model-based selection, but one feature was still missed.
# Running this code also takes significantly longer than that for the model-based selection, because a random forest model is trained 40 times, once for each feature that is dropped.
score = LogisticRegression().fit(X_train_rfe, y_train).score(X_test_rfe, y_test) 
print("Test score: {:.3f}".format(score)) # 0.951
# We can also use the model used inside the RFE to make predictions. This uses only the feature set that was selected.
print("Test score: {:.3f}".format(select.score(X_test, y_test))) # 0.951
# Once we’ve selected the right features, the linear model performs as well as the random forest.
# If you are unsure when selecting what to use as input to your machine learning algorithms, automatic feature selection can be quite helpful.
# It is also great for reducing the amount of features needed—for example, to speed up prediction or to allow for more interpretable models.
# In most real-world cases, applying feature selection is unlikely to provide large gains in performance.
# However, it is still a valuable tool in the toolbox of the feature engineer.

#  Utilizing Expert Knowledge :-
# Adding a feature does not force a machine learning algorithm to use it and even if that feature information turns out to be noninformative for prediction, augmenting the data with this information doesn’t hurt. 
# We’ll now look at one particular case of using expert knowledge—though in this case it might be more rightfully called “common sense.”
# The task we want to solve is to predict for a given time and day how many people will rent a bike in front of Andreas’s house—so he knows if any bikes will be left for him.
# In New York, Citi Bike operates a network of bicycle rental stations with a subscription system.
# The Citi Bike stations are all over the city and provide a convenient way to get around.
citibike=mglearn.datasets.load_citibike()
# Here date and time are not the part of dataset but they are the indexes
print(citibike) # (248, 1)
print(np.array(citibike).shape)
print(citibike.index.values)
plt.figure(figsize=(10,4))
# D = Day  ;  3H = 3Hours
xticks=pd.date_range(start=citibike.index.min(),end=citibike.index.max(),freq='D')
# %a = day  ;  %m = month  ; %d = date
plt.xticks(xticks,xticks.strftime('%a %m-%d'),rotation=90,ha='left')
plt.plot(citibike,linewidth=1)
plt.xlabel("Date") 
plt.ylabel("Rentals")
plt.show()
# The input feature is the date and time—say, 2015-08-01 00:00:00—and the output is the number of rentals 
# A (surprisingly) common way that dates are stored on computers is using POSIX time, which is the number of seconds since January 1970 00:00:00 (aka the beginning of Unix time). 
# extract the target values (number of rentals)
y = citibike.values
# convert to POSIX time by dividing by 10**9
X=citibike.index.astype('int64').values.reshape(-1,1)//10**9
# After dividing the int values by 10**9 we will the exact seconds.
print(citibike.index.astype('int64').values)
print(5//2)  # 2
print(X,'\n',y)
# use the first 184 data points for training, and the rest for testing 
n_train = 184
# function to evaluate and plot a regressor on a given feature set 
def eval_on_features(features, target, regressor):
    # split the given features into a training and a test set     
    X_train, X_test = features[:n_train], features[n_train:]
    # also split the target array
    y_train, y_test = target[:n_train], target[n_train:]
    regressor.fit(X_train, y_train)
    print("Test-set R^2: {:.2f}".format(regressor.score(X_test, y_test)))
    y_pred = regressor.predict(X_test)     
    y_pred_train = regressor.predict(X_train)     
    plt.figure(figsize=(10, 3))
    plt.xticks(range(0, len(X), 8), xticks.strftime("%a %m-%d"), rotation=90,ha="left")
    plt.plot(range(n_train), y_train, label="train")
    plt.plot(range(n_train, len(y_test) + n_train), y_test, '-', label="test")     
    plt.plot(range(n_train), y_pred_train, '--', label="prediction train")
    plt.plot(range(n_train, len(y_test) + n_train), y_pred, '--',label="prediction test")     
    plt.legend(loc=(1.01, 0))     
    plt.xlabel("Date")     
    plt.ylabel("Rentals")
    plt.show()
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0) 
eval_on_features(X, y, regressor) # -0.04
# The predictions on the training set are quite good, as is usual for random forests. However, for the test set, a constant line is predicted. 
# The R2 is –0.04, which means that we learned nothing. 
# The problem lies in the combination of our feature and the random forest.
# The value of the POSIX time feature for the test set is outside of the range of the feature values in the training set.
# Trees and therefore random forests, cannot extrapolate to feature ranges outside the training set.
# The result is that the model simply predicts the target value of the closest point in the training set—which is the last time it observed any data.
# This is where our “expert knowledge” comes in. From looking at the rental figures in the training data, two factors seem to be very important: the time of day and the day of the week.
# So, let’s add these two features. We can’t really learn anything from the POSIX time, so we drop that feature. 
# First, let’s use only the hour of the day. 
X_hour = citibike.index.hour.values.reshape(-1, 1) 
eval_on_features(X_hour, y, regressor) # 0.60
# Now we have same prediction or pattern for each day because the output solely depends on hours.
# Now let’s also add the day of the week 
X_hour_week = np.hstack([citibike.index.dayofweek.values.reshape(-1, 1) ,citibike.index.hour.values.reshape(-1, 1)]) 
print(citibike.index.dayofweek.values)
eval_on_features(X_hour_week, y, regressor) # 0.84
# Now we have a model that captures the periodic behavior by considering the day of week and time of day. 
# It has an R2 of 0.84 and shows pretty good predictive performance.
# What this model likely is learning is the mean number of rentals for each combination of weekday and time.
# This actually does not require a complex model like a random forest, so let’s try with a simpler model, LinearRegression.
from sklearn.linear_model import LinearRegression 
eval_on_features(X_hour_week, y, LinearRegression()) # 0.13
# LinearRegression works much worse and the periodic pattern looks odd. #
# The reason for this is that we encoded day of week and time of day using integers, which are interpreted as continuous variables.
# Therefore, the linear model can only learn a linear function of the time of day—and it learned that later in the day, there are more rentals.
# We can capture this by interpreting the integers as categorical variables, by transforming them using One HotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge 
enc = OneHotEncoder()
print(X_hour_week)
X_hour_week_onehot = enc.fit_transform(X_hour_week).toarray()
# 7 Days of week and 3H freq in day makes 8 hours So, now total features will be 15
print(X_hour_week_onehot.shape) # (248, 15)
eval_on_features(X_hour_week_onehot, y, Ridge()) 
# This gives us a much better match than the continuous feature encoding.
# Now the linear model learns one coefficient for each day of the week and one coefficient for each time of the day.
# That means that the “time of day” pattern is shared over all days of the week, though.
# Using interaction features, we can allow the model to learn one coefficient for each combination of day and time of day.
# Now there will be one more feature used having value 1 made by the combination of 2 True Features.
# So now 3 True features and 117 False features makes prediction.
from sklearn.preprocessing import PolynomialFeatures
# interaction_only=True indicates that there will be no feature made by multiply a feature with itself like square of a feature.
poly_transformer = PolynomialFeatures(degree=2, interaction_only=True,include_bias=False)
X_hour_week_onehot_poly = poly_transformer.fit_transform(X_hour_week_onehot)
print(X_hour_week_onehot_poly.shape) # (248, 120)
print(poly_transformer.get_feature_names())
print(X_hour_week_onehot_poly)
lr = Ridge()
eval_on_features(X_hour_week_onehot_poly, y, lr) # 0.85
# The prediction is still same for every 7 days but it has got much accurate.
# This transformation finally yields a model that performs similarly well to the random forest.
# A big benefit of this model is that it is very clear what is learned: one coefficient for each day and time which is one more than the previous model.
# Earlier, there was one coefficient for each day of the week and one coefficient for each time of the day.
# We can simply plot the coefficients learned by the model, something that would not be possible for the random forest. First, we create feature names for the hour and day features:
hour = ["%02d:00" % i for i in range(0, 24, 3)] 
day = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
h=[9,7,5]
d=[0,1,2,3] 
features =  day + hour
print(hour)
print(features)
# ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', '00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']
f=np.array(h).reshape(-1,1)*np.array(d).reshape(1,-1)
print(f)
features_poly = poly_transformer.get_feature_names(features)
print(len(features_poly)) # 120
features_nonzero = np.array(features_poly)[lr.coef_ != 0] 
coef_nonzero = lr.coef_[lr.coef_ != 0]
print(features_nonzero.shape) # (71,) eliminate features like fri.sat, 3:00.mon etc.
print(coef_nonzero.shape)  # (71, )
# Now we can visualize the coefficients learned by the linear model.
plt.figure(figsize=(15, 2)) 
plt.plot(coef_nonzero, 'o')
plt.xticks(np.arange(len(coef_nonzero)), features_nonzero, rotation=90)
plt.xlabel("Feature name")
plt.ylabel("Feature magnitude")
plt.show()











