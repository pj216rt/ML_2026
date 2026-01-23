import pandas as pd
import numpy as np

#need to add Dexter dataset
datasets = [
    {
        "name": "Gisette",
        "X_train": "assignments/hw03/data/gisette_train.data",
        "y_train": "assignments/hw03/data/gisette_train.labels",
        "X_valid": "assignments/hw03/data/gisette_valid.data",
        "y_valid": "assignments/hw03/data/gisette_valid.labels",
    }
]

lamb = 0.001
iters = 200

#loop over datasets
for d in datasets:
    print(f"{d['name']}")

    X_train = pd.read_csv(d["X_train"], delim_whitespace=True, header=None)
    X_valid = pd.read_csv(d["X_valid"], delim_whitespace=True, header=None)

    y_train = pd.read_csv(d["y_train"], header=None).values.ravel()
    y_valid = pd.read_csv(d["y_valid"], header=None).values.ravel()

    #get N and p
    N, p = X_train.shape

    #standardize
    #find function for here

    #loop over the iterations
    for t in range(iters):
        break


#question 2
#only need some of the datasets in the dict
selected_names = {"Gisette"}
filtered_datasets = list(filter(lambda da: da["name"] in selected_names, datasets))

for d in filtered_datasets:
    print(f"{d['name']}")

    X_train = pd.read_csv(d["X_train"], delim_whitespace=True, header=None)
    X_valid = pd.read_csv(d["X_valid"], delim_whitespace=True, header=None)

    y_train = pd.read_csv(d["y_train"], header=None).values.ravel()
    y_valid = pd.read_csv(d["y_valid"], header=None).values.ravel()

N, p = X_train.shape

#lambda reserved
lamb = 0.001