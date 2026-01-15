import pandas as pd
import numpy as np

df = pd.read_csv('assignments/hw02/data/OnlineNewsPopularity/OnlineNewsPopularity.csv')

print(df.head())

columns_to_drop = ['url', 'timedelta']
df_new = df.drop(columns=columns_to_drop)