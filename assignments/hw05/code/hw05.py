import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import svm

#Q1 parameters.  Gradient Descent Iterations
iters = 200
s = 0.001
eta = 0.005

#get train and test
X_train = pd.read_csv("assignments/hw05/data/gisette_train.data", delim_whitespace=True, header=None)
X_valid = pd.read_csv("assignments/hw05/data/gisette_valid.data", delim_whitespace=True, header=None)
    
y_train = pd.read_csv("assignments/hw05/data/gisette_train.labels", header=None).values.ravel()
y_valid = pd.read_csv("assignments/hw05/data/gisette_valid.labels", header=None).values.ravel()

 #standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_valid)

#add column of 1s to augment w
X_train_tilde = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
X_valid_tilde = np.column_stack([np.ones(len(X_test_scaled)), X_test_scaled])


#part a
#get N and p
N, p = X_train_tilde.shape

#initialize to all 0
w = np.zeros(p)

training_loss = []

for i in range(iters):
    if i % 20 == 0:
        print(i)
    #don't need transpose?
    #y_i*x_i*w
    pred = y_train*(X_train_tilde @ w)
    #the f(x) portion
    fun = np.maximum(0.0, 1.0 - pred)

    #loss function is mean of fun with the penalty term
    #dot product of w.  Don't penalize intercept 
    loss = np.mean(fun) + s*(np.dot(w[1:], w[1:]))

    training_loss.append(loss)

    #gradient
    #select points in form of Boolean vector.  T for being less than 1 and contributing, F for being 1 or greater 
    #and not penalizing the intercept term (w0)
    contrib_points = pred < 1.0
    gradient = (-1.0/N)*(X_train_tilde[contrib_points].T @ y_train[contrib_points])
    gradient[1:] = gradient[1:] + (2.0*s*w[1:]) 

    #update w
    w = w - (eta*gradient)

#make predictions.  Need a sign function
yhat_train = np.sign(X_train_tilde @ w)
yhat_test = np.sign(X_valid_tilde @ w)

#what about if we have yhat_train = 0?

train_misclass = np.mean(yhat_train != y_train)
test_misclass  = np.mean(yhat_test != y_valid)

#plot
plt.figure()
plt.plot(np.arange(1, iters + 1), training_loss)
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title("Primal SVM via Gradient Descent: training loss vs iteration")
plt.grid(True, alpha=0.4)
plt.savefig("assignments/hw05/figures/primal_SVM_training_loss.png", dpi=400, bbox_inches="tight")
plt.close()
#plt.show()

#report the errors
#need to save this as a table.  To do later
print(f"{train_misclass:.4f}\t\t{test_misclass:.4f}")

#part b
#SVM 
#https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC 
#doesn't report number of support vectors
#https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
#does
svm_fit = svm.SVC(
    C=1.0,
    kernel="linear"
)

svm_fit.fit(X_train_scaled, y_train)

#make predictions
yhat_train = svm_fit.predict(X_train_scaled)
yhat_test  = svm_fit.predict(X_test_scaled)

#get error
train_err = np.mean(yhat_train != y_train)
test_err  = np.mean(yhat_test != y_valid)

print(f"Train misclassification: {train_err:.4f}")
print(f"Test misclassification : {test_err:.4f}")

#get number of support vectors
num_sv = svm_fit.n_support_.sum()
print("Number of support vectors:", num_sv)
#report this into a table

#part c, using polynomial kernel, degree 2
svm_poly = svm.SVC(
    kernel="poly",
    degree=2,
    C=1.0
)

svm_poly.fit(X_train_scaled, y_train)

#make predictions
yhat_train = svm_poly.predict(X_train_scaled)
yhat_test  = svm_poly.predict(X_test_scaled)

#get error
train_err = np.mean(yhat_train != y_train)
test_err  = np.mean(yhat_test != y_valid)

print(f"Train misclassification: {train_err:.4f}")
print(f"Test misclassification : {test_err:.4f}")

#get number of support vectors
num_sv = svm_poly.n_support_.sum()
print("Number of support vectors:", num_sv)

#part d, RBF kernel estimators
train_errs = []
test_errs = []

gammas = []
for i in range(1,7):
    temp = 10.0**(-i)
    gammas.append(temp)

#looping over the gamma values
for gamma in gammas:
    svm_brf = svm.SVC(
        kernel="rbf",
        C=1.0,
        gamma=gamma
        )
    
    svm_brf.fit(X_train_scaled, y_train)

    # predictions
    yhat_train = svm_brf.predict(X_train_scaled)
    yhat_test  = svm_brf.predict(X_test_scaled)

    # misclassification errors
    train_err = np.mean(yhat_train != y_train)
    test_err  = np.mean(yhat_test != y_valid)

    train_errs.append(train_err)
    test_errs.append(test_err)

    print(f"gamma={gamma:.1e} | train err={train_err:.4f} | test err={test_err:.4f}")

#plot these gamma values
plt.figure()
plt.semilogx(gammas, train_errs, marker="o", label="Train")
plt.semilogx(gammas, test_errs, marker="s", label="Test")

#can use r to get LATEX symbols
plt.xlabel(r"$\gamma$")
plt.ylabel("Misclassification error")
plt.title("RBF Kernel SVM (C = 1): Misclassification Errors vs $\\gamma$")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.5)

plt.show()