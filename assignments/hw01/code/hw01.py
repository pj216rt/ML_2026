import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#problem 1
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
plt.title("Training and Test Misclassification Rates, Gisette Dataset")
plt.xlabel("Maximum depth")
plt.ylabel("Misclassification error")
plt.legend()
plt.grid(True, alpha=0.3)
#plt.show()

#save this image into the figures folder
plt.savefig("assignments/hw01/figures/gisette_misclassification_p1.png", dpi=400, bbox_inches="tight")
plt.close()

#need to create a dataframe for table:
results_q1 = pd.DataFrame({
    "Depth": range(1,13),
    "Training Misclassification": training_missclass,
    "Validation Misclassification": validate_missclass
})

#find lowest validation error and tree depth
lowest_error = results_q1[results_q1['Validation Misclassification'] == results_q1['Validation Misclassification'].min()]

#save for latex.  Discovered you can use the .to_latex command. 
LATEX_table = lowest_error.to_latex(
    index=False,
    caption="Lowest misclassification error, Gisette Dataset",
    label="tab:tree_misclass_error_p1",
    column_format="ccc"
)

print(LATEX_table)



#problem 2, need to repeat this for satimage, madelon, hill-valley, and covtype data
train_data = pd.read_csv("assignments/hw01/data/X_sat.dat", delim_whitespace=True, header=None)
train_labels = pd.read_csv("assignments/hw01/data/Y_sat.dat", header=None)

validate_data = pd.read_csv("assignments/hw01/data/Xtest_sat.dat", delim_whitespace=True, header=None)
validate_labels = pd.read_csv("assignments/hw01/data/Ytest_sat.dat", header=None)

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
plt.title("Training and Test Misclassification Rates, Satimage Dataset")
plt.xlabel("Maximum depth")
plt.ylabel("Misclassification error")
plt.legend()
plt.grid(True, alpha=0.3)
#plt.show()

#save this image into the figures folder
plt.savefig("assignments/hw01/figures/satimage_misclassification_p2.png", dpi=400, bbox_inches="tight")
plt.close()

#need to create a dataframe for table:
results_q1 = pd.DataFrame({
    "Depth": range(1,13),
    "Training Misclassification": training_missclass,
    "Validation Misclassification": validate_missclass
})

#find lowest validation error and tree depth
lowest_error = results_q1[results_q1['Validation Misclassification'] == results_q1['Validation Misclassification'].min()]

#save for latex.  Discovered you can use the .to_latex command. 
LATEX_table = lowest_error.to_latex(
    index=False,
    caption="Lowest misclassification error: Satimage Dataset",
    label="tab:tree_misclass_error_p2_satimage",
    column_format="ccc"
)

print(LATEX_table)

#madelon data
train_data = pd.read_csv("assignments/hw01/data/madelon_train.data", delim_whitespace=True, header=None)
train_labels = pd.read_csv("assignments/hw01/data/madelon_train.labels", header=None)

validate_data = pd.read_csv("assignments/hw01/data/madelon_valid.data", delim_whitespace=True, header=None)
validate_labels = pd.read_csv("assignments/hw01/data/madelon_valid.labels", header=None)

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
plt.title("Training and Test Misclassification Rates, Satimage Dataset")
plt.xlabel("Maximum depth")
plt.ylabel("Misclassification error")
plt.legend()
plt.grid(True, alpha=0.3)
#plt.show()

#save this image into the figures folder
plt.savefig("assignments/hw01/figures/madelon_misclassification_p2.png", dpi=400, bbox_inches="tight")
plt.close()

#need to create a dataframe for table:
results_q1 = pd.DataFrame({
    "Depth": range(1,13),
    "Training Misclassification": training_missclass,
    "Validation Misclassification": validate_missclass
})

#find lowest validation error and tree depth
lowest_error = results_q1[results_q1['Validation Misclassification'] == results_q1['Validation Misclassification'].min()]

#save for latex.  Discovered you can use the .to_latex command. 
LATEX_table = lowest_error.to_latex(
    index=False,
    caption="Lowest misclassification error: madelon Dataset",
    label="tab:tree_misclass_error_p2_madelon",
    column_format="ccc"
)

print(LATEX_table)

#hill and valley dataset
train_data = pd.read_csv("assignments/hw01/data/X_hv.dat", delim_whitespace=True, header=None)
train_labels = pd.read_csv("assignments/hw01/data/Y_hv.dat", header=None)

validate_data = pd.read_csv("assignments/hw01/data/Xtest_hv.dat", delim_whitespace=True, header=None)
validate_labels = pd.read_csv("assignments/hw01/data/Ytest_hv.dat", header=None)

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
plt.title("Training and Test Misclassification Rates, Satimage Dataset")
plt.xlabel("Maximum depth")
plt.ylabel("Misclassification error")
plt.legend()
plt.grid(True, alpha=0.3)
#plt.show()

#save this image into the figures folder
plt.savefig("assignments/hw01/figures/madelon_misclassification_p2.png", dpi=400, bbox_inches="tight")
plt.close()

#need to create a dataframe for table:
results_q1 = pd.DataFrame({
    "Depth": range(1,13),
    "Training Misclassification": training_missclass,
    "Validation Misclassification": validate_missclass
})

#find lowest validation error and tree depth
lowest_error = results_q1[results_q1['Validation Misclassification'] == results_q1['Validation Misclassification'].min()]

#save for latex.  Discovered you can use the .to_latex command. 
LATEX_table = lowest_error.to_latex(
    index=False,
    caption="Lowest misclassification error: madelon Dataset",
    label="tab:tree_misclass_error_p2_madelon",
    column_format="ccc"
)

print(LATEX_table)


#covtype dataset
#still need to work on this


#Problem 3.  Random Forrests
k_values = [3, 10, 30, 100, 300]