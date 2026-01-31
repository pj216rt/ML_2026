import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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

#Q1 parameters
iters = 100
eta = 0
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

    #get N and p
    N, p = X_train.shape

    #standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_valid)
