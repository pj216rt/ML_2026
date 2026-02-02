import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

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

     #threshold  booleanwise
     w_cop[mask & (np.abs(w_cop) <= lamb)] = 0.0

     return w_cop


#Q1 parameters
iters = 100
eta = 0.0
features = [10, 30, 100, 300, 1000, 3000]

#we're going to be using logspace anyways
lambdas_to_search = np.logspace(-4, 0, 5)
print(lambdas_to_search)

#temporary lambda value
lamb = 0.1

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

    #from notes, eta' can be 1/N
    eta_prime = 1.0/N

    #empty to store the results
    results = []

    #ok so what are we doing here.  We can to find a lambda that gets us the closest to some number
    #of features.  first thing we need to do is loop over the number of possible features.
    #going to be at least 2 loops
    for k in features:
        print(k)
        best_lambda = None
        best_delta = None
        best_selected_vars = None

        #this loops just searches for the best lambda
        for j in lambdas_to_search:
            print(j)

            #initialize w to 0s
            w = np.zeros(p)

            #loop over the number of iterations
            for i in range(iters):
                #get predictor
                z = X_train_tilde @ w

                #build fractional part
                frac = 1.0/(1.0 + np.exp(-z))

                #build y- fractional part
                difference = y_train_converted_01 - frac

                #get the update portion
                update_port = X_train_tilde.T @ difference

                #temporary new omega and do the thresholding
                temp = w + (eta_prime*update_port)
                w = threshold_operator(temp, lamb)
            
            #need to count how many variables were selected
            #don't count itnercept and give a small tolerance
            #check to see if we are close to the desired number of features
            k_selected = int(np.sum(np.abs(w[1:]) > 1e-10))
            delta = abs(k_selected - k)

            if (best_delta is None) or (delta < best_delta):
                best_delta = delta
                best_lambda = lamb
                best_selected_vars = k_selected
    
    #now loop over everything again to time plot and get misclass rates
    w = np.zeros(p)

    start = time.time()
    for t in range(iters):
        z = X_train_tilde @ w
        frac = 1.0 / (1.0 + np.exp(-z))
        difference = (y_train_converted_01 - frac)
        update_port = X_train_tilde.T @ difference
        temp = w + (eta_prime*update_port)
        w = threshold_operator(temp, lamb)

    train_time = time.time() - start

    k_selected = int(np.sum(np.abs(w[1:]) > 1e-10))

    #append to results
    results.append({
        "dataset": name,
        "target_num_features": features,
        
    })