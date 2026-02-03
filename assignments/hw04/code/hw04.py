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
        "y_valid": "assignments/hw04/data/gisette_valid.labels",
        "missclass_1000_out": "assignments/hw04/figures/gisette_misclass_1000.png"
    },
    {
        "name": "Dexter",
        "X_train": "assignments/hw04/data/dexter_train.csv",
        "y_train": "assignments/hw04/data/dexter_train.labels",
        "X_valid": "assignments/hw04/data/dexter_valid.csv",
        "y_valid": "assignments/hw04/data/dexter_valid.labels",
        "missclass_1000_out": "assignments/hw04/figures/dexter_misclass_1000.png"
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
lambdas_to_search = np.logspace(-4, 0, 100)

"""
results = []

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
            #print(j)

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
                w = threshold_operator(temp, j)
            
            #need to count how many variables were selected
            #don't count itnercept and give a small tolerance
            #check to see if we are close to the desired number of features
            k_selected = int(np.sum(np.abs(w[1:]) > 1e-10))
            delta = abs(k_selected - k)

            if (best_delta is None) or (delta < best_delta):
                best_delta = delta
                best_lambda = j
                best_selected_vars = k_selected
    
        #now loop over everything again to time plot and get misclass rates
        w = np.zeros(p)

        #need to store missclass error only when trying to select 1000 features
        misclass_1000 = []

        start = time.time()
        for t in range(iters):
            z = X_train_tilde @ w
            frac = 1.0 / (1.0 + np.exp(-z))
            difference = (y_train_converted_01 - frac)
            update_port = X_train_tilde.T @ difference
            temp = w + (eta_prime*update_port)
            w = threshold_operator(temp, best_lambda)

            #need to track within this iterations loop
            #now how to do that
            if k == 1000:
                z_new = X_train_tilde @ w
                p_train_1 = np.exp(-np.logaddexp(0.0, -z_new))
                p_train_0 = np.exp(-np.logaddexp(0.0, z_new))
                odds_train = p_train_1/p_train_0
                yhat_train = (odds_train > 1.0).astype(int)
                train_err = np.mean(yhat_train != y_train_converted_01)

                #append to misclass_1000 
                misclass_1000.append(train_err)

        train_time = time.time() - start

        k_selected = int(np.sum(np.abs(w[1:]) > 1e-10))

        #train and test misclass errors
        z_train = X_train_tilde @ w
        z_test = X_valid_tilde @ w

        #get probabilities for 1 and 0
        p_train_1 = np.exp(-np.logaddexp(0.0, -z_train))
        p_train_0 = np.exp(-np.logaddexp(0.0, z_train))

        p_test_1 = np.exp(-np.logaddexp(0.0, -z_test))
        p_test_0 = np.exp(-np.logaddexp(0.0, z_test))

        #compute the odds of train and test
        odds_train = p_train_1/p_train_0
        odds_test = p_test_1/p_test_0

        #make predictions.  Predcit 1 if Odds > 1
        #as.int makes things nicer
        yhat_train = (odds_train > 1.0).astype(int)
        yhat_test = (odds_test > 1.0).astype(int)

        #misclassification error
        train_err = np.mean(yhat_train != y_train_converted_01)
        test_err = np.mean(yhat_test != y_valid_converted_01)

        #compute AUC
        test_auc = roc_auc_score(y_valid_converted_01, X_valid_tilde@w)

        #plotting the misclass_error vs iterations for 1000 features
        if k == 1000:
            plt.figure()
            plt.plot(np.arange(1, iters + 1), misclass_1000)
            plt.xlabel("Iteration")
            plt.ylabel("Train misclassification error")
            plt.title(f"{name}: Train misclass vs iteration (k approx 1000)")
            plt.savefig(d["missclass_1000_out"], dpi=400, bbox_inches="tight")
            plt.close()
            #plt.show()

        #append to results
        results.append({
            "dataset": name,
            "target_features": k,
            "selected_features": k_selected,
            "selected_lambda": best_lambda,
            "train_misclass": train_err,
            "test_misclass": test_err,
            "test_auc": test_auc,
            "train_time_sec": train_time
        })

#df for both Gisette and Dexter
df = pd.DataFrame(results)
print(df)

#need to send this table to LATEX
LATEX_table = df.to_latex(
    index=False,
    caption=None,
    label=None,
    column_format="c" * df.shape[1],
    escape=False
)

#save latex table
with open("assignments/hw04/output/misclass_table.tex", "w") as f:
    f.write(LATEX_table)

#plot the final train and test misclassification error vs # of selected features
#using a semilog axis
for name in df["dataset"].unique():
    #create filename to save plot
    outpath = f"assignments/hw04/figures/{name}_misclass_vs_features.png"

    #select the results for a given dataset
    sub = df[df["dataset"] == name].copy()
    
    #works better sorted
    #sub = sub.sort_values("selected_features")

    plt.figure()
    plt.semilogx(sub["selected_features"].to_numpy(), sub["train_misclass"].to_numpy(), marker="o", label="Train")
    plt.semilogx(sub["selected_features"].to_numpy(), sub["test_misclass"].to_numpy(), marker="s", label="Test")

    plt.xlabel("Number of selected features")
    plt.ylabel("Misclassification error")
    plt.title(f"{name}: Final misclassification error vs selected features")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.savefig(outpath, dpi=400, bbox_inches="tight")
    plt.close()
    #plt.show()
"""


#Q2, using same datasets
q2_datasets = q1_datasets

s = 0.001
mu = 100
iters = 100
features = [10, 30, 100, 300, 1000, 3000]

for d in q2_datasets:
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
    
    #need to check that y is +1 or -1.  If loop?
    y_train = pd.read_csv(d["y_train"], header=None).values.ravel()
    y_valid = pd.read_csv(d["y_valid"], header=None).values.ravel()

    #need to force these ya values to be -1/+1
    #https://numpy.org/devdocs/reference/generated/numpy.where.html
    y_train = np.where(y_train > 0, 1.0, -1.0)
    y_valid = np.where(y_valid > 0, 1.0, -1.0)

    #convert labels to 0/1 for likelihood to work
    y_train_converted_01 = (y_train == 1).astype(int)
    y_valid_converted_01 = (y_valid == 1).astype(int)

    #standardize, not forgetting this part this time
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_valid)

    #add column of 1s to augment w
    X_train_tilde = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
    X_valid_tilde = np.column_stack([np.ones(len(X_test_scaled)), X_test_scaled])

    #get N, p
    N, p = X_train_tilde.shape

    #loop over the number of possible features
    for feat in features:
        #initialize to 0
        beta = np.zeros(p)

        start = time.time()

        for i in range(iters):
            #get predictor
            z = X_train_tilde @ beta
            yz = y_train * z

            #need numpy function I found
            #https://numpy.org/devdocs/reference/generated/numpy.log1p.html
            exponent_part = np.exp(-yz)
            log_part = np.log1p(exponent_part)
            loss = np.mean(log_part)

            #no sparsity penalty
            #compute gradient
            part1 = 1.0 / (1.0 + np.exp(yz))
            grad = -(1/N)*(X_train_tilde.T @ (y_train*part1))

            #update Beta using gradient step
            #is s equivalent to eta in the notes??  It  has to be
            beta = beta - (s*grad)

            #inverse scheduler
            denom = (2.0*i*mu) + iters
            numer = (iters - (2*i))
            ratio = numer/denom

            #get maximum of 0 or the ratio.  Exclude the intercept
            p_feat = p-1
            max_portion = max(0, ratio)
            Mi = np.round(feat + (p-feat)*max_portion)
            Mi = int(Mi)

            #need to keep the top Mi variables, not including intercept
            #sort.  Keep the largest Mi
            b = beta[1:]
            abs_b = abs(b)
            sort = np.argsort(abs_b)
            
        end = time.time()
        run_time = end-start
            
