import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

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
#https://stackoverflow.com/questions/26984414/efficiently-sorting-a-numpy-array-in-descending-order
#mean descrease in impurity

MDI_22 = rf_22.feature_importances_
MDI_500 = rf_500.feature_importances_

#permutation stuff
PERM_22 = permutation_importance(rf_22, X_valid, y_valid, n_repeats=10, n_jobs=4)
PERM_500 = permutation_importance(rf_500, X_valid, y_valid, n_repeats=10, n_jobs=4)

#need just the importances mean
PERM_22  = PERM_22.importances_mean
PERM_500 = PERM_500.importances_mean

#need to sort and take the first 50.  
MDI_22_sorted = -np.sort(-MDI_22)[:50]
MDI_500_sorted = -np.sort(-MDI_500)[:50]
PERM_22_sorted = -np.sort(-PERM_22)[:50]
PERM_500_sorted = -np.sort(-PERM_500)[:50]

ranks = np.arange(1, 51)

plt.figure()
plt.plot(ranks, MDI_22_sorted, marker="o", linewidth=1, label="RF, 22 MDI")
plt.plot(ranks, MDI_500_sorted, marker="v", linewidth=1, label="RF, 500 MDI")
plt.plot(ranks, PERM_22_sorted, marker="1", linewidth=1, label="RF, 22 Permutation")
plt.plot(ranks, PERM_500_sorted, marker="*", linewidth=1, label="RF, 500 Permutation")

#Axis is a little confusing bc we're overlaying various models (RFs of different feature sizes)
#and two different methods of determining feature importance
plt.xlabel("Rank among top 50 features, within each model and method")
plt.ylabel("Feature importance (MDI or mean accuracy decrease)")
plt.title("Top 50 Feature Importances: MDI vs Permutation")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("assignments/hw06/figures/feature_importances.png", dpi=400, bbox_inches="tight")
plt.close()
#plt.show()

#part c
k_values = [10, 20, 30, 40]