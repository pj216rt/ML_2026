import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

#need to add Dexter dataset
datasets = [
    {
        "name": "Gisette",
        "X_train": "assignments/hw03/data/gisette_train.data",
        "y_train": "assignments/hw03/data/gisette_train.labels",
        "X_valid": "assignments/hw03/data/gisette_valid.data",
        "y_valid": "assignments/hw03/data/gisette_valid.labels"
    },
    {
        "name": "Madelon",
        "X_train": "assignments/hw03/data/madelon_train.data",
        "y_train": "assignments/hw03/data/madelon_train.labels",
        "X_valid": "assignments/hw03/data/madelon_valid.data",
        "y_valid": "assignments/hw03/data/madelon_valid.labels"
    }
]

lamb = 0.001
iters = 200

#need to define the loss function

#Q1
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
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_valid)

    #add column of 1s
    X_train_tilde = np.column_stack([np.ones(len(X_train)), X_train])
    X_valid_tilde = np.column_stack([np.ones(len(X_valid)), X_valid])

    #initialize w and create empty vector to store loss values
    w = np.zeros(p+1)
    losses = []
    
    print("here")
    #loop over the iterations
    for t in range(iters):
        z = X_train_tilde @ w 

        #compute log likelihood
        #z = linear algebra product of X and w, with column of ones added
        L = np.sum(y_train*z - np.log(1.0 + np.exp(z)))

        #Loss function.  Don't penalize the intercept
        C = ((-1.0/N)*L) + (lamb*np.dot(w[1:], w[1:]))

        #Gradient of log-likelihood


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
    
    #get shape
    N, p = X_train.shape
    
    #add column of 1s
    X_train_tilde = np.column_stack([np.ones(len(X_train)), X_train])
    X_valid_tilde = np.column_stack([np.ones(len(X_valid)),  X_valid])

    #identity matrix needed for analytical solution
    I = np.eye(p+1)

    #matrix math
    LHS = X_train_tilde.T @ X_train_tilde + (lamb*N*I)
    RHS = X_train_tilde.T @ y_train

    #analytical minimizer of the penalized squared loss function
    w_hat = np.linalg.solve(LHS, RHS)

    #predictions make using w^T @ x
    train_score = X_train_tilde @ w_hat
    test_score = X_valid_tilde @ w_hat

    #need to get sign function
    yhat_train = np.sign(train_score)
    yhat_test  = np.sign(test_score)

    # misclassification error
    train_misclass = np.mean(yhat_train != y_train)
    test_misclass  = np.mean(yhat_test != y_valid)

    print(f"Train misclassification error: {train_misclass:.4f}")
    print(f"Test misclassification error:  {test_misclass:.4f}")

    #get this into a latex table?
    results_df = pd.DataFrame([{
        "lambda": lamb,
        "train_misclass_error": train_misclass,
        "test_misclass_error": test_misclass,
    }])

    LATEX_table = results_df.to_latex(
    index=False,
    caption="Train and Test Misclassification Errors",
    label="tab:train_test_missclass_errors_hw03",
    column_format="c" * results_df.shape[1],
    escape=False
    )
    
    print(LATEX_table)

    #ROC curves
    #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, train_score)
    fpr_test, tpr_test, thresholds_test = roc_curve(y_valid, test_score)

    #need to find a way to plot this and the result from Q1
    plt.figure()
    plt.plot(fpr_train, tpr_train, label="Train ROC")
    plt.plot(fpr_test, tpr_test, label="Test ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{d['name']}: Ridge (squared loss) ROC")
    plt.legend()
    plt.show()

#can we augment w?