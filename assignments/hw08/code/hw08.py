import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

#functions to impliment the various accuracy measures
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cluster.contingency_matrix.html
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
def conting_matrix(observed, predicted):
    #function to compute the contingency matrix
    print("Contingency Matrix:")

    conting_matrix_results = contingency_matrix(observed, predicted)

    #need to maximize, so we need to minimize the negative of the contingency matrix
    row_ind, col_ind = linear_sum_assignment(-conting_matrix_results)

    #then matching cluster labels with true labels

#function to apply the adjusted rand index to the observed and predicted labels
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
def applying_rand_index(observed, predicted):
    results = adjusted_rand_score(observed, predicted)

    return results

#part a
def make_dataset_part_a(a):
    XQ = multivariate_normal.rvs(
        mean=[0.0, 0.0],
        cov=[[9.0, 0.0], [0.0, 9.0]],
        size=500
    )
    YQ = np.zeros(500)

    mu_a = np.array([a, 0.0])
    XA = multivariate_normal.rvs(
        mean=mu_a,
        cov=np.eye(2),
        size=500
        )
    YA = np.ones(500)

    #merge
    X = np.vstack([XQ, XA])
    y = np.concatenate([YQ, YA])

    return X, y

#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
#https://python-course.eu/machine-learning/expectation-maximization-and-gaussian-mixture-models-gmm.php
#https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture
for a in range(5):
    #get dataset
    X, y = make_dataset_part_a(a=a)

    #K-means clustering.  Adding a random seed this time.
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=1000+a)
    kmeans_labels = kmeans.fit_predict(X)

    #EM clustering
    EM_cluster = GaussianMixture(n_components=2, n_init=10, random_state=1000+a)
    EM_cluster_labels = EM_cluster.fit_predict(X)

    #the adjusted rand inex
    rand_index_kmeans = applying_rand_index(y, kmeans_labels)
    rand_index_EM = applying_rand_index(y, EM_cluster_labels)