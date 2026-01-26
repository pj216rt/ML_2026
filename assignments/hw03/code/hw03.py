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
    },
    {
        "name": "Dexter",
        "X_train": "assignments/hw03/data/dexter_train.data",
        "y_train": "assignments/hw03/data/dexter_train.labels"
    }
]

lamb = 0.001
iters = 200
eta_vals = [1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

#things to find the best eta value
best_eta = None
best_final = np.inf
best_losses = None
best_w = None

#Q1
#loop over datasets
for d in datasets:
    #get name of dataset
    name = d["name"]
    print(name)

    #do something else for the Dexter set
    if d["name"] == "Dexter":
        break

    #load in datasets
    X_train = pd.read_csv(d["X_train"], delim_whitespace=True, header=None)
    X_valid = pd.read_csv(d["X_valid"], delim_whitespace=True, header=None)

    y_train = pd.read_csv(d["y_train"], header=None).values.ravel()
    y_valid = pd.read_csv(d["y_valid"], header=None).values.ravel()

    #convert to 0/1
    y_train_converted_01 = (y_train == 1).astype(int)
    y_valid_converted_01 = (y_valid == 1).astype(int)

    #get N and p
    N, p = X_train.shape

    #standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_valid)

    #add column of 1s
    X_train_tilde = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
    X_valid_tilde = np.column_stack([np.ones(len(X_test_scaled)), X_test_scaled])

    #store all loss curves for this dataset
    loss_curves = {}
    best_eta = None
    best_final = np.inf
    best_w = None

    #loop over the eta values
    for eta in eta_vals:
        w = np.zeros(p+1)
        losses = []
        prev = np.inf
        monotone = True
        
        print(f"{eta}")
        #loop over the iterations
        for t in range(iters):
            #get predictor
            z = X_train_tilde @ w 
            #compute log likelihood
            #z = linear algebra product of X and w, with column of ones added
            #https://numpy.org/devdocs/reference/generated/numpy.logaddexp.html
            L = np.sum(y_train_converted_01*z - np.logaddexp(0.0, z))
            
            #Loss function.  
            #C = ((-1.0/N)*L) + (lamb*np.dot(w[1:], w[1:]))
            C = ((-1.0/N)*L) + (lamb*np.dot(w, w))
            losses.append(C)
            
            #checking to see if the loss is monotone decreasing
            if t > 0 and C > prev + 1e-12:
                  monotone = False
                  break
            prev = C
            
            exponential_term = np.exp(-np.logaddexp(0.0, -z))
            gradient_L = X_train_tilde.T @ (y_train_converted_01 - exponential_term)
            
            #update w
            w = w - eta*lamb*w + (eta/N)*gradient_L

        #check if the series of loss scores is monotone decreasing and that the
        #length is correct
        if monotone and len(losses) == iters:
            losses = np.array(losses)
            loss_curves[eta] = losses
            
            if losses[-1] < best_final:
                best_final = losses[-1]
                best_eta = eta
                best_w = w.copy()

    print(f"Best eta for {name}: {best_eta} (final loss {best_final:.6f})")
    
    plt.figure()
    
    for eta, curve in loss_curves.items():
        if eta == best_eta:
            plt.plot(np.arange(1, iters + 1), curve, linewidth=3,
                     alpha=1.0, label=rf"$\eta={eta}$ (best)")
        else:
            plt.plot(np.arange(1, iters + 1), curve, linewidth=1.2,
                     alpha=0.35, label=rf"$\eta={eta}$")
            
    plt.xlabel("Iteration")
    plt.ylabel("Training loss $C(w)$")
    plt.title(rf"{name} Training Loss, different $\eta$")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


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