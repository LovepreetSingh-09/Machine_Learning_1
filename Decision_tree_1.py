# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 00:39:03 2019

@author: user
"""
# Descision Tree

import graphviz
import mglearn
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
from IPython.display import display
from sklearn.model_selection import train_test_split

# Bescision Tree Classifier

mglearn.plots.plot_animal_tree()
plt.show()
# There are two common strategies to prevent overfitting: stopping the creation of the tree early (also called pre-pruning), or building the tree but then removing or collapsing nodes that contain little information (also called post-pruning or just pruning).
# Possible criteria for pre-pruning include limiting the maximum depth of the tree, limiting the maximum number of leaves, or requiring a minimum number of points in a node to keep splitting it.
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=42)
dtc=DecisionTreeClassifier().fit(X_train,y_train)
# Training set score: 1.000 and Test set score: 0.923
# This makes that tree is complex and makes it overfit
print("Training set score: {:.3f}".format(dtc.score(X_train, y_train)))
print("Test set score: {:.3f}".format(dtc.score(X_test, y_test)))
# To prevent the tree from arbitarily deep and complex, we need to define the max depth
tree=DecisionTreeClassifier(max_depth=4,random_state=0).fit(X_train,y_train)
print("Training set score: {:.3f}".format(tree.score(X_train, y_train)))
# Training set score: 0.988 and Test set score: 0.951
print("Test set score: {:.3f}".format(tree.score(X_test, y_test)))
# We can visualize the tree using the export_graphviz function from the tree module. 
# This writes a file in the .dot file format, which is a text file format for storing graphs.
# We set an option to color the nodes to reflect the majority class in each node and pass the class and features names so the tree can be properly labeled 
from sklearn.tree import export_graphviz
export_graphviz(tree,out_file='tree.dot',class_names=['milignant','bemign'],feature_names=cancer.feature_names,impurity=False,filled=True)
with open('tree.dot') as f:
    dot_graph=f.read()
# print(dot_graph) 
# dot_graph is a string file which is stored in the tree.dot file
display(graphviz.Source(dot_graph))
plt.show()

# Print the features importance which are mostly used and always sum of all the feature importance = 1
print('Feature Importances:',tree.feature_importances_)
def plot_f_i_cancer(model):
    n_features=cancer.data.shape[1]
    plt.barh(range(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features),cancer.feature_names)
    # np.arange(n_features) means the list 0 to 29 representing each feature by the next term in the bracket
    print(np.arange(n_features))
    plt.xlabel('Features Importance')
    plt.ylabel('Features')
    plt.ylim(-1,n_features)  
    plt.show()
plot_f_i_cancer(tree)
plt.show()

# output class is not monotonous, means we cannot say “a high value of X[1] means class 0, and a low value means class 1” (or vice versa).
tree=mglearn.plots.plot_tree_not_monotone()
display(tree)
plt.show()

# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
ram_prices=pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH,'ram_price.csv'))
print(ram_prices)
plt.semilogy(ram_prices.date,ram_prices.price)
plt.xlabel('Year')
plt.ylabel('Prices in $/Mbyte')
plt.show()

data_train=ram_prices[ram_prices.date<2000]
data_test=ram_prices[ram_prices.date>=2000]
X_train=data_train.date[:,np.newaxis]
# we use a log-transform to get a simpler relationship of data to target 
y_train=np.log(data_train.price)
tree=DecisionTreeRegressor().fit(X_train,y_train)
lireg=LinearRegression().fit(X_train,y_train)
X_all=ram_prices.date[:,np.newaxis]
# print(ram_prices.date[:])
# np.newaxis makes a new column of date of all the rows while upper command simply shows the values of date
print(ram_prices.date[:,np.newaxis])
pred_tree=tree.predict(X_all)
pred_lireg=lireg.predict(X_all)
# undo log transformation
price_tree=np.exp(pred_tree)
price_lireg=np.exp(pred_lireg)
plt.semilogy(data_train.date,data_train.price,label='Training data')
plt.semilogy(data_test.date,data_test.price,label='Test data')
# Tree prediction doesn't predict the test data. It just simply keep on predicting the last known point or it just makes a straight horizontal line from the last point
# . The tree has no ability to generate “new” responses, outside of what was seen in the training data. This shortcoming applies to all models based on trees. 
plt.semilogy(ram_prices.date,price_tree,label='Tree Prediction')
plt.semilogy(ram_prices.date,price_lireg,label='Linear Prediction')
plt.legend()
plt.show()

# Ensembles are methods that combine multiple machine learning models to create more powerful models. 
# There are two ensemble models that are effective on a wide range of datasets for classification and regression, both of which use decision trees as their building blocks: random forests and gradient boosted decision trees.





