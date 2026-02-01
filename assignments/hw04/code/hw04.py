import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import time

#Q1 datasets
q1_datasets = [
    {
        "name": "Gisette",
        "X_train": "assignments/hw04/data/gisette_train.data",
        "y_train": "assignments/hw04/data/gisette_train.labels",
        "X_valid": "assignments/hw04/data/gisette_valid.data",
        "y_valid": "assignments/hw04/data/gisette_valid.labels"
    },
    {
        "name": "Dexter",
        "X_train": "assignments/hw04/data/dexter_train.csv",
        "y_train": "assignments/hw04/data/dexter_train.labels",
        "X_valid": "assignments/hw04/data/dexter_valid.csv",
        "y_valid": "assignments/hw04/data/dexter_valid.labels"
    }
]

#thresholding function, w is vector of length p+1
#I don't think we should penalize the intercept
#https://stats.stackexchange.com/questions/559416/ridge-regression-subtlety-on-intercept
#will element wise decide whether or not to select variable
def threshold_operator(ws, lamb):
     #make copy of w vector
     w_cop = ws.copy()

     #https://numpy.org/devdocs/reference/generated/numpy.ones_like.html
     #plus don't do anything to the intercept
     #We should just get a boolean vector (F, T, T, T, T, etc)
     mask = np.ones_like(w_cop, dtype=bool)
     mask[0] = False

     #threshold


#Q1 parameters
iters = 100
eta = 0.0
eta_prime = 0.001
features = [10, 30, 100, 300, 1000, 3000]

#Q1
for d in q1_datasets:
    #get name of dataset
    name = d["name"]
    print(name)

    #Load datasets.  use default read.csv if Dexter dataset
    if name == "Dexter":
        X_train = pd.read_csv(d["X_train"], header=None)
        X_valid = pd.read_csv(d["X_valid"], header=None)
    else:
        X_train = pd.read_csv(d["X_train"], delim_whitespace=True, header=None)
        X_valid = pd.read_csv(d["X_valid"], delim_whitespace=True, header=None)
    
    y_train = pd.read_csv(d["y_train"], header=None).values.ravel()
    y_valid = pd.read_csv(d["y_valid"], header=None).values.ravel()

    #convert labels to 0/1 for likelihood to work
    y_train_converted_01 = (y_train == 1).astype(int)
    y_valid_converted_01 = (y_valid == 1).astype(int)

    #standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_valid)

    #add column of 1s to augment w
    X_train_tilde = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
    X_valid_tilde = np.column_stack([np.ones(len(X_test_scaled)), X_test_scaled])

    #get N, p
    N, p = X_train_tilde.shape

    #initialize w to 0s
    w = np.zeros(p)

    for t in range(iters):
            #get predictor
            z = X_train_tilde @ w

            #build fractional part
            frac = 1.0/(1.0 + np.exp(-z))

            #build y- fractional part
            difference = y_train_converted_01 - frac

            #get the update portion
            update_port = X_train_tilde.T @ difference

            #temporary new omega
            temp = w + (eta_prime*update_port)
