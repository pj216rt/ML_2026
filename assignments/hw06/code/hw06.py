import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

#working with the Madelon Data
#loading in everything.  Reusing the copy of the dataset from hw01
X_train = pd.read_csv("assignments/hw01/data/madelon_train.data", delim_whitespace=True, header=None)
X_valid = pd.read_csv("assignments/hw01/data/madelon_valid.data", delim_whitespace=True, header=None)

y_train = pd.read_csv("assignments/hw01/data/madelon_train.labels", header=None).values.ravel()
y_valid = pd.read_csv("assignments/hw01/data/madelon_valid.labels", header=None).values.ravel()

#part a.  Need two random forests and need to fit them
rf_22 = RandomForestClassifier(n_estimators=100, max_features=22)
rf_500 = RandomForestClassifier(n_estimators=100, max_features=500)

#fitting
rf_22.fit(X_train, y_train)
rf_500.fit(X_train, y_train)

#making predictions
rf_22_pred_train = rf_22.predict(X_train)
rf_500_pred_train = rf_500.predict(X_train)

rf_22_pred_test = rf_22.predict(X_valid)
rf_500_pred_test = rf_500.predict(X_valid)

#errors
rf_22_train_error = 1.0 - accuracy_score(y_train, rf_22_pred_train)
rf_500_train_error = 1.0 - accuracy_score(y_train, rf_500_pred_train)

rf_22_test_error = 1.0 - accuracy_score(y_valid, rf_22_pred_test)
rf_500_test_error = 1.0 - accuracy_score(y_valid, rf_500_pred_test)

#need to put this in a table
part1 = pd.DataFrame({
    "Model": [
        "Random Forest, 22 Features",
        "Random Forest, 500 Features"
    ],
    "Train Misclass. Error": [
        rf_22_train_error, rf_500_train_error
    ],
    "Test Misclass. Error": [
        rf_22_test_error, rf_500_test_error
    ]
})

print(part1)

#need to send this table to LATEX
LATEX_table = part1.to_latex(
    index=False,
    caption=None,
    label=None,
    column_format="c" * part1.shape[1],
    escape=False
)

#save latex table
with open("assignments/hw06/output/train_test_errors_a.tex", "w") as f:
    f.write(LATEX_table)

#part b
#https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
#mean descrease in impurity
start_time = time.time()
importances = rf_22.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_22.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")