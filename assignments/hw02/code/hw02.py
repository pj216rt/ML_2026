import pandas as pd
import numpy as np

#reading in the csv file
df = pd.read_csv('assignments/hw02/data/OnlineNewsPopularity/OnlineNewsPopularity.csv')

#dropping the first two columns by name.  space in timedelta
columns_to_drop = ['url', ' timedelta']
df_new = df.drop(columns=columns_to_drop)

print(df_new.head())

#selecting all but the last column as the predictors
X = df_new.iloc[:,:-1]

#selecting the last column as response
Y = np.log(df_new.iloc[:,-1])

