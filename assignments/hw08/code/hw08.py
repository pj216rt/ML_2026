import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    conting_matrix_results = contingency_matrix(observed, predicted)

    #need to maximize, so we need to minimize the negative of the contingency matrix.
    #returns the rolw and column indices for the optimal assignment.
    row_ind, col_ind = linear_sum_assignment(-conting_matrix_results)

    #maximum diag sum/total number of samples
    accuracy = conting_matrix_results[row_ind, col_ind].sum()/np.sum(conting_matrix_results)

    return accuracy

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

#need a way to store the accuracy of each run for both K-means and EM
acc_kmeans_list  = []
acc_em_list  = []

rand_kmeans_list = []
rand_em_list = []

a_list = []

#loop over the different a values
for a in range(5):
    for run in range(10):
        #get dataset
        X, y = make_dataset_part_a(a=a)

        #K-means clustering.  Adding a random seed this time. for each run and each dataset
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=1000+a+run)
        kmeans_labels = kmeans.fit_predict(X)

        #EM clustering
        EM_cluster = GaussianMixture(n_components=2, n_init=10, random_state=1000+a+run)
        EM_cluster_labels = EM_cluster.fit_predict(X)

        #computing accuracy
        accuracy_kmeans = conting_matrix(y, kmeans_labels)
        accuracy_EM = conting_matrix(y, EM_cluster_labels)

        #append to lists
        acc_kmeans_list.append(accuracy_kmeans)
        acc_em_list.append(accuracy_EM)

        #the adjusted rand inex
        rand_index_kmeans = applying_rand_index(y, kmeans_labels)
        rand_index_EM = applying_rand_index(y, EM_cluster_labels)

        #append the rand_index to the list
        rand_kmeans_list.append(rand_index_kmeans)
        rand_em_list.append(rand_index_EM)

        a_list.append(a)

    #plot the clustering results if a==0 and some random run number.  9 here.
    #https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
    if a == 0 and run == 9:
        #set up figure
        fig, axes = plt.subplots(1, 2)

        #K-means plot
        axes[0].scatter(X[:, 0], X[:, 1],
            c=kmeans_labels,
            s=10,alpha=0.7
            )
        axes[0].set_title("K-means Clustering (a = 0)")
        axes[0].set_xlabel("X1")
        axes[0].set_ylabel("X2")

        #EM plot
        axes[1].scatter(X[:, 0], X[:, 1],
            c=EM_cluster_labels,
            s=10, alpha=0.7
            )
        axes[1].set_title("EM Clustering (a = 0)")
        axes[1].set_xlabel("X1")
        axes[1].set_ylabel("X2")

        plt.tight_layout()
        plt.savefig("assignments/hw08/figures/cluster_results_a.png", dpi=400, bbox_inches="tight")
        plt.close()
        #plt.show()

#little hard to see, maybe add a jitter?
#plot the accuracy of each run vs as a seperate dots on the same plot
#different colors for K-means and EM
plt.figure()
plt.scatter(a_list, np.array(acc_kmeans_list), label = "K-means")
plt.scatter(a_list, np.array(acc_em_list), label = "EM")
plt.xlabel("Dataset a")
plt.ylabel("Accuracy")
plt.title("Accuracy across 5 datasets, 10 runs each")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("assignments/hw08/figures/accuracy_vs_a.png", dpi=400, bbox_inches="tight")
plt.close()
#plt.show()

#plotting the adjusted rand index now.  Same as above but with the rand index instead of accuracy
plt.figure()
plt.scatter(a_list, np.array(rand_kmeans_list), label = "K-means")
plt.scatter(a_list, np.array(rand_em_list), label = "EM")
plt.xlabel("Dataset a")
plt.ylabel("Adjusted Rand Index")
plt.title("Adjusted Rand Index across 5 datasets, 10 runs each")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("assignments/hw08/figures/ari_vs_a.png", dpi=400, bbox_inches="tight")
plt.close()
#plt.show()

#need function for KL divergence between two Gaussians
def kl_divergence_gaussians(mu1, cov1, mu2, cov2):
    #get dimension
    d = mu1.shape[0]


#part b
for i in range(10):
    #each element is from a standard normal distibubtion
    #https://numpy.org/doc/2.1/reference/random/generated/numpy.random.normal.html
    M = np.random.normal(0, 1, size=(2, 2))
    
    #SVD decomposition
    U, S, Vh = np.linalg.svd(M)

    #create Sigma matrix from U
    D = np.diag([25,1])
    Sigma = U @ D @ U.T

    #generate X centered at 0 with covariance Sigma
    XQ = multivariate_normal.rvs(
        mean=[0.0, 0.0],
        cov=Sigma,
        size=500
    )

    #generate XP from (10,0) with covariance Sigma
    XP = multivariate_normal.rvs(
        mean=[10.0, 0.0],
        cov=Sigma,
        size=500
    )