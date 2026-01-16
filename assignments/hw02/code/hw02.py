import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import Pipeline

#reading in the csv file
df = pd.read_csv('assignments/hw02/data/OnlineNewsPopularity/OnlineNewsPopularity.csv')

#dropping the first two columns by name.  space in timedelta
columns_to_drop = ['url', ' timedelta']
df_new = df.drop(columns=columns_to_drop)

#selecting all but the last column as the predictors
X = df_new.iloc[:,:-1]

#selecting the last column as response
Y = np.log(df_new.iloc[:,-1])

#null model, simply using y_bar to predict y
#set up cross validation
cv = KFold(n_splits=5, random_state=1, shuffle=True)

#using DummyRegressor always predicting the mean mean of the training set
null_model = DummyRegressor(strategy="mean")

results = cross_validate(null_model, X=X, y=Y, cv=cv, scoring="neg_mean_squared_error", return_train_score=True)

#cross validate uses the negative mean squared error.  Need to negative the negative
mean_train_mse = -results["train_score"].mean()
mean_test_mse = -results["test_score"].mean()

print(mean_train_mse)
print(mean_test_mse)

#OLS ridge regression
lambdas = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

#need to loop over the lambdas

#proof of concept from skilearn documentation
#for i, (train_index, test_index) in enumerate(cv.split(X)):
    #print(f"Fold {i}:")
    #print(f"  Train: index={train_index}")
    #print(f"  Test:  index={test_index}")

#OLS regression with backwards elimination
#https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html

#maybe pipelines? Avoid leaking test set to the train set?
#https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
k_values = range(5, 30, 5)
results = []

for k in k_values:
    print(k)
    

#random forest regression
depths = range(1, 11)
rows = []

#loop over depths
for d in depths:
    print(d)
    #getting idx of the train and test amongst the splits
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = Y.iloc[train_idx], Y.iloc[test_idx]

        #turning on the oob_score
        rf = RandomForestRegressor(n_estimators=100, max_depth=d, oob_score=True, random_state=1234)
        
        #fitting stuff
        rf.fit(X_train, y_train)

        #make predictions
        y_pred_train = rf.predict(X_train)
        y_pred_test = rf.predict(X_test)

        #results
        #using the oob_prediction_ to compute MSE
        results = {
            "depth": d,
            "train_mse": mean_squared_error(y_train, y_pred_train),
            "train_R2": r2_score(y_train, y_pred_train),
            "oob_R2": rf.oob_score_,
            "oob_MSE": mean_squared_error(y_train, rf.oob_prediction_),
            "test_mse": mean_squared_error(y_test, y_pred_test),
            "test_R2": r2_score(y_test, y_pred_test)
        }

        rows.append(results)

#making table and plotting
table1 = pd.DataFrame(rows)

#need to get means and standard deviations
#using agg function
summary_table = table1.groupby("depth").agg(
    train_r2_mean=("train_R2", "mean"),
    train_r2_sd=("train_R2", "std"),
    oob_r2_mean=("oob_R2", "mean"),
    oob_r2_sd=("oob_R2", "std"),
    test_r2_mean=("test_R2", "mean"),
    test_r2_sd=("test_R2", "std"),
    train_mse_mean=("train_mse", "mean"),
    train_mse_sd=("train_mse", "std"),
    oob_mse_mean=("oob_MSE", "mean"),
    oob_mse_sd=("oob_MSE", "std"),
    test_mse_mean=("test_mse", "mean"),
    test_mse_sd=("test_mse", "std"),
)

print(summary_table)

#needed to convert to np array
plt.figure()
plt.plot(summary_table.index.to_numpy(), summary_table["train_r2_mean"].to_numpy(), label="Training R2")
plt.plot(summary_table.index.to_numpy(), summary_table["oob_r2_mean"].to_numpy(), label="OOB R2")
plt.plot(summary_table.index.to_numpy(), summary_table["test_r2_mean"].to_numpy(), label="Test R2")

plt.xlabel("Maximum Tree Depth")
plt.ylabel("Average R2")
plt.legend()
plt.tight_layout()
plt.show()