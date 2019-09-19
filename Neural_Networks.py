# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 01:08:19 2019

@author: user
"""
# Neural Network or MLPs (multilayer preceptrons)
import graphviz
import mglearn
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
from IPython.display import display
from sklearn.model_selection import train_test_split

# ŷ = w[0] * x[0] + w[1] * x[1] + ... + w[p] * x[p] + b
# ŷ is a weighted sum of the input features x[0] to x[p] , weighted by the learned coefficients w[0] to w[p] .
display(mglearn.plots.plot_logistic_regression_graph())

# Hidden units represents an intermediate processing step, which are again combined using weighted sums to yield the final result 
display(mglearn.plots.plot_single_hidden_layer_graph())

# After computing a weighted sum for each hidden unit, a nonlinear function is applied to the result—usually the rectifying nonlinearity ( also known as rectified linear unit or relu) or the tangens hyperbolicus (tanh).
# The result of this function is then used in the weighted sum that computes the output, ŷ.
# The rectified linear unit or relu cuts off values below zero. Used as np.maximum(x,0) in the model 
# The tangens hyperbolicus (tanh) saturates to –1 for low input values and +1 for high input values. Used as np.tanh(x) in the model.
# Either nonlinear function allows the neural network to learn much more complicated functions than a linear model could.
line=np.linspace(-3,3,100)
plt.plot(line,np.tanh(line),label='tanh')
plt.plot(line,np.maximum(line,0),label='relu')
plt.xlabel('x')
plt.ylabel('relu(x),tanh(x)')
plt.legend(loc='best')
plt.show()

# For the small neural network, the formula for computing ŷ in the case of regression would be (when using a tanh nonlinearity):
# h[0] = tanh(w[0, 0] * x[0] + w[1, 0] * x[1] + w[2, 0] * x[2] + w[3, 0] * x[3] + b[0]) 
# h[1] = tanh(w[0, 1] * x[0] + w[1, 1] * x[1] + w[2, 1] * x[2] + w[3, 1] * x[3] + b[1]) 
# h[2] = tanh(w[0, 2] * x[0] + w[1, 2] * x[1] + w[2, 2] * x[2] + w[3, 2] * x[3] + b[2]) 
# ŷ = v[0] * h[0] + v[1] * h[1] + v[2] * h[2] + b

# MLPClassifier :-
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=42)
mlp=MLPClassifier(solver='lbfgs',random_state=0).fit(X_train,y_train)
mglearn.plots.plot_2d_separator(mlp,X_train,fill=True,alpha=0.3)
mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
plt.xlabel("Feature 0") 
plt.ylabel("Feature 1")
plt.show()
# By default, MLPClassifier has 100 hidden units and default non-linearity = relu
# For this small dataset we will reduce the no. of hidden units to reduce the complexity of the model
mlp=MLPClassifier(solver='lbfgs',random_state=0,hidden_layer_sizes=[10]).fit(X_train,y_train)
mglearn.plots.plot_2d_separator(mlp,X_train,fill=True,alpha=0.3)
mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
plt.xlabel("Feature 0") 
plt.ylabel("Feature 1")
plt.show()
# Now we will use the two hidden layers
mlp=MLPClassifier(solver='lbfgs',random_state=0,hidden_layer_sizes=[10,10]).fit(X_train,y_train)
mglearn.plots.plot_2d_separator(mlp,X_train,fill=True,alpha=0.3)
mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
plt.xlabel("Feature 0") 
plt.ylabel("Feature 1")
plt.show()
# Now use the tanh non-linearity
mlp=MLPClassifier(solver='lbfgs',random_state=0,activation='tanh',hidden_layer_sizes=[10,10]).fit(X_train,y_train)
mglearn.plots.plot_2d_separator(mlp,X_train,fill=True,alpha=0.3)
mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
plt.xlabel("Feature 0") 
plt.ylabel("Feature 1")
plt.show()

# L2 penality can also be applied in the MLPClassifier to shrink the weights or slopes toward zero
# The parameter for this is alpha and by default it has been set to very little regularization
fig,axes=plt.subplots(2,4,figsize=(20,8))
for axx,n_hidden_nodes in zip(axes,[10,100]):
    for ax,alpha in zip(axx,[0.0001,0.01,0.1,1]):
        mlp=MLPClassifier(solver='lbfgs',random_state=0,hidden_layer_sizes=[n_hidden_nodes,n_hidden_nodes],alpha=alpha)
        mlp.fit(X_train,y_train)
        mglearn.plots.plot_2d_separator(mlp,X_train,fill=True,alpha=0.4,ax=ax)
        mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train,ax=ax)
        ax.set_title('n_hidden=[{},{}]\nalpha={}'.format(n_hidden_nodes,n_hidden_nodes,alpha))
plt.show()

# An important property of neural networks is that their weights are set randomly before learning is started, and this random initialization affects the model that is learned.
# That means that even when using exactly the same parameters, we can obtain very different models when using different random seeds.
# If the networks are large, and their complexity is chosen properly, this should not affect accuracy too much, but it is worth keeping in mind (particularly for smaller networks). 
fig,axes=plt.subplots(2,4,figsize=(20,8))
for i,ax in enumerate(axes.ravel()):
        mlp=MLPClassifier(solver='lbfgs',random_state=i,hidden_layer_sizes=[100,100])
        mlp.fit(X_train,y_train)
        mglearn.plots.plot_2d_separator(mlp,X_train,fill=True,alpha=0.4,ax=ax)
        mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train,ax=ax)
        ax.set_title('random_seed={}'.format(i))
plt.show()

from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
print('Cancer dataset maximas:\n',cancer.data.max(axis=0))
print('Features :\n',cancer.keys())
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
mlp = MLPClassifier(random_state=42) 
mlp.fit(X_train, y_train)
print("Accuracy on training set: {:.2f}".format(mlp.score(X_train, y_train))) # Accuracy on training set: 0.94
print("Accuracy on test set: {:.2f}".format(mlp.score(X_test, y_test)))  # Accuracy on test set: 0.92
# The accuracy is good but not as good as other models Because of scaling of data
# compute the mean value per feature on the training set
mean_on_train=X_train.mean(axis=0)
# compute the standard deviation of each feature on the training 
std_on_train=X_train.std(axis=0)
# subtract the mean, and scale by inverse standard deviation
# afterward, mean=0 and std=1
X_train_scaled=(X_train-mean_on_train)/std_on_train
print("X_train_scaled:\n",X_train_scaled)
print("X_train_scaled std :\n",X_train_scaled.std(axis=0))
print("X_train_scaled mean :\n",X_train_scaled.mean(axis=0)) # The mean will not be exactly 0 but x 10^(-15/-16) power will be there 
# THE SAME transformation (using training mean and std) on the test set
X_test_scaled = (X_test - mean_on_train) / std_on_train
mlp = MLPClassifier(random_state=0)
mlp.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled, y_train))) # Accuracy on training set: 0.991
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test))) # Accuracy on test set: 0.965''''
# Now after scaling the accuracy scores are much better but we have got a warning about the max no. of iterarions has been reached 
mlp = MLPClassifier(max_iter=1000,random_state=0)
mlp.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled, y_train))) # Accuracy on training set:  1.000
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))  # Accuracy on test set:  0.972
# by max_iter the model becomes overfit
# There is some gap between training and test scores. To reduce it, we need to make the model a little bit more complex
# For Complexity define alpha, by default alpha = 0.0001. So, increase the alpha
mlp = MLPClassifier(max_iter=1000,random_state=0)
mlp.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled, y_train))) # Accuracy on training set: 0.988
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test))) # Accuracy on test set: 0.972

# The following plot shows the weights that were learned connecting the input to the first hidden layer.
# The rows in this plot correspond to the 30 input features
# The columns correspond to the 100 hidden units.
# Light colors represent large positive values, while dark colors represent negative values.
plt.figure(figsize=(20,5))
print(mlp.coefs_[0].shape) # (30,100)
# mlp.coefs[0] is an array of wights where rows are 30 input features and columns are 100 hidden units
# mlp.coef[1] is the array of weights from hidden units to the output node y and its shape is (100,1)
plt.imshow(mlp.coefs_[0],cmap='viridis')
plt.yticks(range(cancer.feature_names.size),cancer.feature_names)
plt.xlabel("Columns in weight matrix / Hidden Units")
plt.ylabel("Input feature") 
plt.colorbar()
plt.show()
# One possible inference we can make is that features that have very small weights for all of the hidden units are “less important” to the model. 
# There is also the question of how to learn the model, or the algorithm that is used for learning the parameters, which is set using the algorithm parameter.
# There are two easy-to-use choices for algorithms :-
# 1.) The default is 'adam', which works well in most situations but is quite sensitive to the scaling of the data (so it is important to always scale your data to 0 mean and unit variance).
# 2.) The other one is 'lbfgs', which is quite robust but might take a long time on larger models or larger datasets.
# -> There is also the more advanced 'sgd' option, which is what many deep learning researchers use. The 'sgd' option comes with many additional param‐eters that need to be tuned for best results.




