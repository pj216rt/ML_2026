import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#Q1 datasets
#need to change these to hw05
q1_datasets = [
    {
        "name": "Gisette",
        "X_train": "assignments/hw04/data/gisette_train.data",
        "y_train": "assignments/hw04/data/gisette_train.labels",
        "X_valid": "assignments/hw04/data/gisette_valid.data",
        "y_valid": "assignments/hw04/data/gisette_valid.labels"
    }
]

#Q1 parameters.  Gradient Descent Iterations
iters = 200
s = 0.001

for d in q1_datasets:
    name = d["name"]
    print(name)

    #get train and test
    X_train = pd.read_csv(d["X_train"], delim_whitespace=True, header=None)
    X_valid = pd.read_csv(d["X_valid"], delim_whitespace=True, header=None)
    
    y_train = pd.read_csv(d["y_train"], header=None).values.ravel()
    y_valid = pd.read_csv(d["y_valid"], header=None).values.ravel()

    #standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_valid)

    #add column of 1s to augment w
    X_train_tilde = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
    X_valid_tilde = np.column_stack([np.ones(len(X_test_scaled)), X_test_scaled])