import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#loading in the data

#file was seperated via spaces.  We know that its dimensions are 6000 rows and 5000 columns.  
#needed to use delim_whitespace instead of sep, and specify no header
train_data = pd.read_csv("assignments/hw01/data/gisette_train.data", delim_whitespace=True, header=None)
train_labels = pd.read_csv("assignments/hw01/data/gisette_train.labels", header=None)

validate_data = pd.read_csv("assignments/hw01/data/gisette_valid.data", delim_whitespace=True, header=None)
validate_labels = pd.read_csv("assignments/hw01/data/gisette_valid.labels", header=None)

#now we can use it to make predictions on the train and test (really validate)

#need to loop from 1 to 12
#initialize the empty storage arrays
training_missclass = []
validate_missclass = []

#12 depths
for i in range(1, 13):
    #fitting the decision treee with max depth of i
    dtree = tree.DecisionTreeClassifier(max_depth=i)
    dtree = dtree.fit(train_data,train_labels)

    #predict
    training_predict = dtree.predict(train_data)
    validate_predict = dtree.predict(validate_data)

    #get misclassification rate
    #check accuracy scores.  Returns fraction of correctly classified sampler.  I think that this means that the
    # #misclassification rate is simply 1- this value
    training_missclass_rate = 1 - accuracy_score(train_labels, training_predict)
    validation_missclass_rate = 1 - accuracy_score(validate_labels, validate_predict)

    #append to arrays
    training_missclass.append(training_missclass_rate)
    validate_missclass.append(validation_missclass_rate)

#need to plot things now
plt.figure()
plt.plot(range(1,13), training_missclass, marker='o', label='Training error')
plt.plot(range(1,13), validate_missclass, marker='o', label='Validation error')
plt.title("Training and Test Misclassification Rates")
plt.subtitle("Gisette Dataset")
plt.xlabel("Maximum depth")
plt.ylabel("Misclassification error")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#save this image into the figures folder