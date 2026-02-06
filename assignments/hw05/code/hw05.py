import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#Q1 parameters.  Gradient Descent Iterations
iters = 200
s = 0.001

#get train and test
X_train = pd.read_csv("assignments/hw05/data/gisette_train.data", delim_whitespace=True, header=None)
X_valid = pd.read_csv("assignments/hw05/data/gisette_valid.data", delim_whitespace=True, header=None)
    
y_train = pd.read_csv("assignments/hw05/data/gisette_train.labels", header=None).values.ravel()
y_valid = pd.read_csv("assignments/hw05/data/gisette_valid.labels", header=None).values.ravel()

 #standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_valid)

#add column of 1s to augment w
X_train_tilde = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
X_valid_tilde = np.column_stack([np.ones(len(X_test_scaled)), X_test_scaled])

#get N and p
N, p = X_train_tilde.shape

#initialize to all 0
w = np.zeros(p)

training_loss = []

for i in range(iters):
    #don't need transpose?
    #y_i*x_i*w
    pred = y_train*(X_train_tilde @ w)
    #the f(x) portion
    fun = np.maximum(0.0, 1.0 - pred)

    #loss function is mean of fun with the penalty term
    #dot product of w
    loss = np.mean(fun) + s*(np.dot(w,w))

    training_loss.append(loss)