import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyRegressor

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