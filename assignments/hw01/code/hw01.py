import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

#need a double loop.  Outer loop over datasets, inner loop over depts
depths = range(1, 13)

#creating a dict to help loop over the datasets more
datasets = [
    {
        "name": "Gisette",
        "X_train": "assignments/hw01/data/gisette_train.data",
        "y_train": "assignments/hw01/data/gisette_train.labels",
        "X_valid": "assignments/hw01/data/gisette_valid.data",
        "y_valid": "assignments/hw01/data/gisette_valid.labels",
        "fig_out": "assignments/hw01/figures/gisette_misclassification_error.png",
        "latex_caption": "Lowest misclassification error, Gisette Dataset",
        "latex_label": "tab:gissette_tree_misclass_error"
    },
    {
        "name": "Satimage",
        "X_train": "assignments/hw01/data/X_sat.dat",
        "y_train": "assignments/hw01/data/Y_sat.dat",
        "X_valid": "assignments/hw01/data/Xtest_sat.dat",
        "y_valid": "assignments/hw01/data/Ytest_sat.dat",
        "fig_out": "assignments/hw01/figures/satimage_misclassification_error.png",
        "latex_caption": "Lowest misclassification error: Satimage Dataset",
        "latex_label": "tab:Satimage_tree_misclass_error"
    },
    {
        "name": "Madelon",
        "X_train": "assignments/hw01/data/madelon_train.data",
        "y_train": "assignments/hw01/data/madelon_train.labels",
        "X_valid": "assignments/hw01/data/madelon_valid.data",
        "y_valid": "assignments/hw01/data/madelon_valid.labels",
         "fig_out": "assignments/hw01/figures/madelon_misclassification_error.png",
        "latex_caption": "Lowest misclassification error: Madelon Dataset",
        "latex_label": "tab:Madelon_tree_misclass_error"
    },
    {
        "name": "Hill-Valley",
        "X_train": "assignments/hw01/data/X_hv.dat",
        "y_train": "assignments/hw01/data/Y_hv.dat",
        "X_valid": "assignments/hw01/data/Xtest_hv.dat",
        "y_valid": "assignments/hw01/data/Ytest_hv.dat",
        "fig_out": "assignments/hw01/figures/hill_valley_misclassification_error.png",
        "latex_caption": "Lowest misclassification error: Hill-Valley Dataset",
        "latex_label": "tab:hill_valley_tree_misclass_error"
    },
    {
        "name": "Covtype",
        "X_train": "assignments/hw01/data/covtype.data",
        "fig_out": "assignments/hw01/figures/Covtype_misclassification_error.png",
        "latex_caption": "Lowest misclassification error: Covtype Dataset",
        "latex_label": "tab:Covtype_tree_misclass_error"
    }
]

#parts a and b
for i in datasets:
    #special case for the Covtype dataset.
    if i["name"] == "Covtype":
        data = pd.read_csv(i["X_train"], header=None).to_numpy()
        
        X = data[:, :-1]
        y = data[:, -1]

        X_train = X[:15120]
        y_train = y[:15120]

        X_valid = X[15120:]
        y_valid = y[15120:]
    else:
        #reading in dataset, if dataset is not covtype
        X_train = pd.read_csv(i["X_train"], delim_whitespace=True, header=None)
        X_valid = pd.read_csv(i["X_valid"], delim_whitespace=True, header=None)

        y_train = pd.read_csv(i["y_train"], header=None)
        y_valid = pd.read_csv(i["y_valid"], header=None)

    #initialize empty arrays
    train_err = []
    valid_err = []

    #inner loop looping over the tree depths
    for d in depths:
        #fit tree
        dec_tree = tree.DecisionTreeClassifier(max_depth=d)
        dec_tree = dec_tree.fit(X_train, y_train)

        #make predictions
        training_predictions = dec_tree.predict(X_train)
        validation_predictions = dec_tree.predict(X_valid)

        #compute error
        training_error = 1 - accuracy_score(y_train, training_predictions)
        validation_error = 1 - accuracy_score(y_valid, validation_predictions)

        train_err.append(training_error)
        valid_err.append(validation_error)
    
    #now need to plot this
    plt.figure()
    plt.plot(list(depths), train_err, marker="o", label="Training error")
    plt.plot(list(depths), valid_err, marker="o", label="Validation error")
    plt.title(f"Training and Test Misclassification Rates, {i['name']} Dataset")
    plt.xlabel("Maximum depth")
    plt.ylabel("Misclassification error")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(i["fig_out"], dpi=400, bbox_inches="tight")
    plt.close()

    #need a table to find the best depth
    results = pd.DataFrame({
        "Depth": list(depths),
        "Training Misclassification": train_err,
        "Validation Misclassification": valid_err
    })

    #find lowest validation error and tree depth
    lowest_error = results[results['Validation Misclassification'] == results['Validation Misclassification'].min()]

    #save as latex table:
    LATEX_table = lowest_error.to_latex(
    index=False,
    caption=i["latex_caption"],
    label=i["latex_label"],
    column_format="ccc"
    )

    #print
    print(LATEX_table)


#parts c and d
k_values = [3, 10, 30, 100, 300]

#only need some of the datasets in the dict
selected_names = {"Gisette", "Satimage", "Madelon", "Hill-Valley"}
filtered_datasets = list(filter(lambda da: da["name"] in selected_names, datasets))

#loop over filter datasets for parts c and d:
for data in filtered_datasets:
    print(data["name"])

    #get data
    X_train = pd.read_csv(data["X_train"], delim_whitespace=True, header=None)
    X_valid = pd.read_csv(data["X_valid"], delim_whitespace=True, header=None)

    #got warning about y being a column vector.  Python recomended using ravel
    y_train = pd.read_csv(data["y_train"], header=None).values.ravel()
    y_valid = pd.read_csv(data["y_valid"], header=None).values.ravel()

    train_errors = []
    test_errors = []

    #iterate over k values
    for k in k_values:
        forest = RandomForestClassifier(n_estimators=k, max_features=71)
        forest = forest.fit(X_train, y_train)

        #making predictions
        prediction_training = forest.predict(X_train)
        prediction_validation = forest.predict(X_valid)

        #error
        train_error = 1 - accuracy_score(y_train, prediction_training)
        valid_error = 1 - accuracy_score(y_valid, prediction_validation)

        #append to array
        train_errors.append(train_error)
        test_errors.append(valid_error)