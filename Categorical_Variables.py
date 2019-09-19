# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 21:16:29 2019

@author: user
"""

# Categorical features (Discrete Features) 
# To represent your data best for a particular application is known as feature engineering

import graphviz
import mglearn
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
from IPython.display import display
from sklearn.model_selection import train_test_split

# One-Hot_Encoding or One-Out-Of-N-Encoding or Dummy Variables 
# The idea behind dummy variables is to replace a categorical variable with one or more new features that can have the values 0 and 1.
# The values 0 and 1 make sense in the formula for linear binary classification (and for all other models in scikit-learn) and we can represent any number of categories by introducing one new feature per category.

# Using Pandas :-
# The file has no headers naming the columns, so we pass header=None
# And provide the column names explicitly in "names"
adult_path=os.path.join(mglearn.datasets.DATA_PATH,'adult.data')
data=pd.read_csv(adult_path,header=None,index_col=False,names=['age', 'workclass', 'fnlwgt', 'education',  'education-num','marital-status', 'occupation', 'relationship', 'race', 'gender','capital-gain', 'capital-loss', 'hours-per-week', 'native-country','income'])
# For illustration purposes, we only select some of the columns 
data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week','occupation', 'income']]
# IPython.display allows nice output formatting within the Jupyter notebook 
display(data.head())
print(data.gender.value_counts())
print('Original Features : ',list(data.columns))
# pandas automatically converts the catgorical Features into Dummy Features and didn't touch the numeric features or variables
data_dummies=pd.get_dummies(data)
print('Features after get_dummies : ',list(data_dummies.columns))
# You can see that the continuous features age and hours-per-week were not touched, while the categorical features were expanded into one new feature for each possible value.
# Every Workclass, Education, Gender, Occupation, income (<=50K and >50K) Value has converted into a new feature.
print(data_dummies.head())
# Pandas include the last value while slicing unlike numpy 
features=data_dummies.loc[:,'age':'occupation_ Transport-moving']
X=features.values
y=data_dummies['income_ >50K']
print('X_Shape : {}\nY_Shape : {}'.format(X.shape,y.shape)) # X_Shape : (32561, 44)  Y_Shape : (32561,)
from sklearn.linear_model import LogisticRegression 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logreg = LogisticRegression() 
logreg.fit(X_train, y_train)
print("Test score: {:.2f}".format(logreg.score(X_test, y_test))) # Test score: 0.81

# Here, the categorical variables were encoded as strings.
# On the one hand, it opens up the possibility of spelling errors, but on the other hand, it clearly marks a variable as categorical.
# But Often, whether for ease of storage or because of the way the data is collected, categorical variables can be encoded as integers. 
# Pandas treats all treats all numeric features as continuous.
# So, we can either use scikit-learn's OneHotEncoder or convert that whole feature or column into string.
# create a DataFrame with an integer feature and a categorical string feature
demo_df = pd.DataFrame({'Integer Feature': [0, 1, 2, 1] , 'Categorical Feature': ['socks', 'fox', 'socks', 'box']}) 
display(demo_df)
# Converts only Categorical Feature
display(pd.get_dummies(demo_df))
# For converting Integer Feature into Categorical
demo_df['Integer Feature']=demo_df['Integer Feature'].astype(str)
display(pd.get_dummies(demo_df))

# Binning, Discretization, Linear Models, and Trees :- 
# The best way to represent data depends not only on the semantics of the data, but also on the kind of model you are using.
# Linear models and tree-based models two large and very commonly used families, have very different properties when it comes to how they work with different feature representations. 
# Lets compare these 2 models on wave regression datset which has only 1 input feature.
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor
X, y = mglearn.datasets.make_wave(n_samples=100) 
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
reg = DecisionTreeRegressor(min_samples_split=3).fit(X, y) 
plt.plot(line, reg.predict(line), label="decision tree")
reg = LinearRegression().fit(X, y)
plt.plot(line, reg.predict(line), label="linear regression")
plt.plot(X[:, 0], y, 'o', c='k') 
plt.ylabel("Regression output") 
plt.xlabel("Input feature") 
plt.legend(loc="best")
plt.show()
# Descretization or Binning :-
# One way to make linear models more powerful on continuous data is to use binning (also known as discretization) of the feature to split it up into multiple features
bins=np.linspace(-3,3,11)
print(bins)
which_bin=np.digitize(X,bins=bins)
print(X[:5],'\n',which_bin[:5])
print(which_bin.shape) # (100, 1)
# Use OneHotEncoder to transform which_bin into X_binned
# X_binned will have 10 features for each bin
from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder(sparse=False)
# encoder.fit finds the unique values that appear in which_bin
encoder.fit(which_bin)
# transform creates the one-hot encoding
X_binned=encoder.transform(which_bin)
print(X_binned.shape) # (100, 10)
line_binned=encoder.transform(np.digitize(line,bins=bins))
reg = DecisionTreeRegressor(min_samples_split=3).fit(X_binned, y) 
plt.plot(line, reg.predict(line_binned), label="decision tree")
reg = LinearRegression().fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label="linear regression")
plt.plot(X[:, 0], y, 'o', c='k') 
plt.vlines(bins, -3, 3, linewidth=1, alpha=.2)
plt.ylabel("Regression output") 
plt.xlabel("Input feature") 
plt.legend(loc="best")
plt.show()
# Here both algos predct the same as both lines in the graph are over each other
# For each bin there is same and constant prediction
# Here, linear model gets much more flexible while Decision tree gets less flexible
# Binning features generally has no beneficial effect for tree-based models, as these models can learn to split up the data anywhere. 
# Additionally, decision trees look at multiple features at once, while binning is usually done on a per-feature basis.
# The reasons to use a linear model for a particular dataset is due to very large and high-dimensional so, binning can be a great way to increase modeling power.

# Interactions and Polynomials :-
# Another way to enrich a feature representation, particularly for linear models, is adding interaction features and polynomial features of the original data. 
# To make a slope for each bin, we have to add original feature again into the binned features making 11 features
X_combined=np.hstack([X_binned,X])
line_combined=np.hstack([line_binned,line])
print(X_combined,'\n',X_combined.shape) #  (100, 11)
reg = DecisionTreeRegressor(min_samples_split=3).fit(X_combined, y) 
plt.plot(line, reg.predict(line_combined), label="decision tree")
reg = LinearRegression().fit(X_combined, y)
plt.plot(line, reg.predict(line_combined), label="linear regression")
plt.plot(X[:, 0], y, 'o', c='k') 
plt.vlines(bins, -3, 3, linewidth=1, alpha=.2)
plt.ylabel("Regression output") 
plt.xlabel("Input feature") 
plt.legend(loc="best")
plt.show()
# Here, Decision tree has been backed to its original more flexible predictions and linear model has slope which is downward for each bin.
# This downward slope for each bin is due to single x-axis feature which has single slope and it is shared by each bin
# Interaction:-
# To make different slope for each bin we will add an interaction or product feature that indicates which bin a data point is in and where it lies on the x-axis.
# This feature is a product of the bin indicator and the original feature.
X_product=np.hstack([X_binned,X*X_binned])
line_product=np.hstack([line_binned,line*line_binned])
print(X_product.shape) # (100, 20)
reg = LinearRegression().fit(X_product, y)
plt.plot(line, reg.predict(line_product), label="linear regression")
plt.plot(X[:, 0], y, 'o', c='k') 
plt.vlines(bins, -3, 3, linewidth=1, alpha=.2)
plt.ylabel("Regression output") 
plt.xlabel("Input feature") 
plt.legend(loc="best")
plt.show()
# Polynomial :-
# Another way to expand features is to use polynomials of the original features.
# For a given feature x, we might want to consider x ** 2, x ** 3, x ** 4, and so on. 
# Then we dont need any binning of data for smooth boundary of predictions.
# This is implemented in PolynomialFeatures in the preprocessing module.
from sklearn.preprocessing import PolynomialFeatures
# include polynomials up to x ** 10:
# the default "include_bias=True" adds a feature that's constantly 1
poly=PolynomialFeatures(degree=10,include_bias=False)
poly.fit(X)
X_poly=poly.transform(X)
# print(X_poly,'\n',X_poly.shape) # (100, 10)
print(poly.get_feature_names()) # ['x0', 'x0^2', 'x0^3', 'x0^4', 'x0^5', 'x0^6', 'x0^7', 'x0^8', 'x0^9', 'x0^10']
line_poly=poly.transform(line)
reg = LinearRegression().fit(X_poly, y)
plt.plot(line, reg.predict(line_poly), label="linear regression")
plt.plot(X[:, 0], y, 'o', c='k') 
plt.vlines(bins, -3, 3, linewidth=1, alpha=.2)
plt.ylabel("Regression output") 
plt.xlabel("Input feature") 
plt.legend(loc="best")
plt.show()
# As you can see, polynomial features yield a very smooth fit on this one-dimensional data. 
# As a comparison, here is a kernel SVM model learned on the original data, without any transformation 
from sklearn.svm import SVR
for gamma in [1, 10]:     
    svr = SVR(gamma=gamma).fit(X, y)
    plt.plot(line, svr.predict(line), label='SVR gamma={}'.format(gamma))
plt.plot(X[:, 0], y, 'o', c='k') 
plt.ylabel("Regression output") 
plt.xlabel("Input feature") 
plt.legend(loc="best")
plt.show()
# Here, default kernel is rbf which makes a smooth boundary without any transformation or adding features.
from sklearn.datasets import load_boston 
from sklearn.preprocessing import MinMaxScaler
boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)
# rescale data 
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
poly = PolynomialFeatures(degree=2).fit(X_train_scaled)
X_train_poly = poly.transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
print("X_train.shape: {}".format(X_train.shape)) # (379, 13)
print("X_train_poly.shape: {}".format(X_train_poly.shape)) # (379, 105)
print("Polynomial feature names:\n{}".format(poly.get_feature_names())) 
# 78 features of combination of 2 features ; 13 original features ; 13 square of original features and a constant feature 1
# Thus, makes 105 features.
from sklearn.linear_model import Ridge 
ridge = Ridge(alpha=0.01).fit(X_train_scaled, y_train) 
print("Score without interactions: {:.3f}".format(ridge.score(X_test_scaled, y_test))) #  0.635
ridge = Ridge(alpha=0.25).fit(X_train_poly, y_train) 
print("Score with interactions: {:.3f}".format(ridge.score(X_test_poly, y_test))) # 0.775
# Clearly, the interactions and polynomial features gave us a good boost in performance when using Ridge.
# When using a more complex model like a random forest, the story is a bit different.
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100).fit(X_train_scaled, y_train)
print("Score without interactions: {:.3f}".format(rf.score(X_test_scaled, y_test))) # 0.796
rf = RandomForestRegressor(n_estimators=100).fit(X_train_poly, y_train) 
print("Score with interactions: {:.3f}".format(rf.score(X_test_poly, y_test))) # 0.776
# You can see that even without additional features, the random forest beats the performance of Ridge. 
# Adding interactions and polynomials actually decreases performance slightly.

# Univariate Non-linear Transformation :-
# Sometimes, the mathematical functions like log, sin, exp etc. can help to transform features.
# linear models and neural networks are very tied to the scale and distribution of each feature, and if there is a nonlinear relation between the feature and the target, that becomes hard to model, particularly in regression.
# The functions log and exp can help by adjusting the relative scales in the data so that they can be captured better by a linear model or neural network. 
rnd=np.random.RandomState(0)
X_org=rnd.normal(size=(1000,3))
w=rnd.normal(size=3)
print(X_org,w)
X=rnd.poisson(10*np.exp(X_org))
print(X.shape,10*np.exp(X_org)) # (1000, 3)
y=np.dot(X_org,w)
print(y.shape) # (1000,)
# No. of Feature Appearances
print(np.bincount(X[:,0]))
print(np.where(X[:,0]==0)) # Gives the index where the value is 0
bins=np.bincount(X[:,0])
plt.bar(range(len(bins)),bins)
plt.ylabel("Number of appearances") 
plt.xlabel("Value")
plt.show()
from sklearn.linear_model import Ridge
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
score = Ridge().fit(X_train, y_train).score(X_test, y_test) 
print("Test score: {:.3f}".format(score))  # 0.622
# As you can see from the relatively low R2 score, Ridge was not able to really capture the relationship between X and y.
# Applying a logarithmic transformation can help here.
# Because the value 0 appears in the data (and the logarithm is not defined at 0), we can’t actually just apply log, but we have to compute log(X + 1).
X_train_log=np.log(X_train+1)
X_test_log=np.log(X_test+1)
plt.hist(X_train_log[:, 0], bins=25, color='gray')
plt.ylabel("Number of appearances") 
plt.xlabel("Value")
plt.show()
score = Ridge().fit(X_train_log, y_train).score(X_test_log, y_test) 
print("Test score: {:.3f}".format(score)) # 0.875
# These kinds of transformations are irrelevant for tree-based models but might be essential for linear models. 
# Sometimes it is also a good idea to transform the target variable y in regression. 
# Trying to predict counts (say, number of orders) is a fairly common task, and using the log(y + 1) transformation often helps. 
# Binning, polynomials and interactions can have a huge influence on how models perform on a given dataset.
# This is particularly true for less complex models like linear models and naive Bayes models.
# Tree-based models, on the other hand, are often able to discover important interactions themselves and don’t require transforming the data explicitly most of the time. 
# Other models, like SVMs, nearest neighbors and neural networks, might sometimes benefit from using binning, interactions, or polynomials, but the implications there are usually much less clear than in the case of linear models.










