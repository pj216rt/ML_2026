import pandas as pd
import numpy as np
from sklearn import tree
#import sklearn

#loading in the data

#file was seperated via spaces.  We know that its dimensions are 6000 rows and 5000 columns.  
#needed to use delim_whitespace instead of sep, and specify no header
train_data = pd.read_csv("assignments/hw01/data/gisette_train.data", delim_whitespace=True, header=None)
train_labels = pd.read_csv("assignments/hw01/data/gisette_train.labels", header=None)

validate_data = pd.read_csv("assignments/hw01/data/gisette_valid.data", delim_whitespace=True, header=None)
validate_labels = pd.read_csv("assignments/hw01/data/gisette_valid.labels", header=None)

#good here
#print(train_data.shape)

#needed to update sklearn
#print(np.__version__)
#print(sklearn.__version__)

#setting up and fitting the tree
dtree = tree.DecisionTreeClassifier(max_depth=5)
dtree = dtree.fit(train_data,train_labels)

#now we can use it to make predictions
training_predict = dtree.predict(train_data)
validate_predict = dtree.predict(validate_data)