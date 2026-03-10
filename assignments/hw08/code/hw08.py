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

#need function for KL divergence between two Gaussians
#I think there may be an extra 2 in the KL formula provided
#https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/
def kl_divergence_gaussians(mu1, cov1, mu2, cov2):
    #get dimension
    d = mu1.shape[0]

    #need the log determinant of the covariance matrices
    #https://numpy.org/doc/2.1/reference/generated/numpy.linalg.slogdet.html
    sign1, logdet1 = np.linalg.slogdet(cov1)
    sign2, logdet2 = np.linalg.slogdet(cov2)

    #trace term
    t_term = np.trace(np.linalg.inv(cov2) @ cov1)

    #quadtratid term
    q_term = (mu2 - mu1).T @ np.linalg.inv(cov2) @ (mu2 - mu1)

    #two 1/2 terms.  make sure I have everything
    kl_div = 0.5 * ((logdet2 - logdet1 - d) + t_term + q_term)

    return kl_div

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

random_run = np.random.randint(0, 10)

#loop over the different a values
#just using the default K-means implementation from sklearn for this part
for a in range(5):
    for run in range(10):
        #get dataset
        X, y = make_dataset_part_a(a=a)

        #K-means clustering.  Adding a random seed this time. for each run and each dataset
        kmeans = KMeans(n_clusters=2, n_init=1, random_state=1000+a+run)
        kmeans_labels = kmeans.fit_predict(X)

        #EM clustering
        EM_cluster = GaussianMixture(n_components=2, n_init=1, random_state=1000+a+run)
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
    if a == 0 and run == random_run:
        #set up figure
        fig, axes = plt.subplots(1, 2, figsize=(12,5))

        #K-means plot
        axes[0].scatter(X[:, 0], X[:, 1],
            c=kmeans_labels,
            s=10,alpha=0.7
            )
        axes[0].set_title(f"K-means Clustering (a = 0, run = {run})")
        axes[0].set_xlabel("X1")
        axes[0].set_ylabel("X2")

        #EM plot
        axes[1].scatter(X[:, 0], X[:, 1],
            c=EM_cluster_labels,
            s=10, alpha=0.7
            )
        axes[1].set_title(f"EM Clustering (a = 0, run = {run})")
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

#function to impliment k-means clustering from the slides, which cares about the covariance structure
#k-means with either no covariance modelling or full covariance modelling
def k_means_clustering(X, n_clusters, max_iters=100, seed=123,
                       covariance="identity"):
    
    #set random seed and get dimensions
    np.random.seed(seed)
    n, d = X.shape

    #get initial centers by randomly sampling from the data points without replacement
    #https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
    idx = np.random.choice(n, n_clusters, replace=False)
    centers = X[idx]

    #initialize covariance matrices and labels.  Set labels to -1 so they are not initialized to any cluster
    Sigmas = np.tile(np.eye(d), (n_clusters, 1, 1))
    labels = -np.ones(n, dtype=int)

    for i in range(max_iters):
        old_labels = labels.copy()

        #compute distances
        dists = np.zeros((n, n_clusters))

        #if branch for covariance type.  
        if covariance == "identity":
            for k in range(n_clusters):
                dists[:, k] = np.linalg.norm(X - centers[k], axis=1)**2

        elif covariance == "full":
            for k in range(n_clusters):
                diff = X - centers[k]
                invS = np.linalg.inv(Sigmas[k])
                dists[:, k] = np.sum(diff @ invS * diff, axis=1)

        #assign clusters
        #https://numpy.org/devdocs/reference/generated/numpy.argmin.html
        labels = np.argmin(dists, axis=1)

        #stop if labels don't change, even if we haven't hit max_iters
        if np.array_equal(labels, old_labels):
            break

        #update centers and covariances for each group
        for k in range(n_clusters):

            #select the points in the current cluster
            cluster_points = X[labels == k]

            #compute the new mean/centroid
            centers[k] = cluster_points.mean(axis=0)

            #compute the covariance matrix for the cluster if we are using full covariance
            if covariance == "full":
                Sigmas[k] = np.cov(cluster_points, rowvar=False)

            #otherwise, just use an identity matrix
            else:
                Sigmas[k] = np.eye(d)

    return labels, centers, Sigmas

#part b
results = []
plot_runs = []

for i in range(10):
    #each element is from a standard normal distibubtion
    #https://numpy.org/doc/2.1/reference/random/generated/numpy.random.normal.html
    M = np.random.normal(0, 1, size=(2, 2))
    
    #SVD decomposition
    #U rotates the data
    U, S, Vh = np.linalg.svd(M)

    #create Sigma matrix from U
    #each run produces a rotated matrix with different variances along the two dimensions
    D = np.diag([25,1])
    Sigma = U @ D @ U.T

    #generate X centered at 0 with covariance Sigma plus labels
    XQ = multivariate_normal.rvs(
        mean=[0.0, 0.0],
        cov=Sigma,
        size=500
    )
    YQ = np.zeros(500)

    #generate XP from (10,0) with covariance Sigma
    XP = multivariate_normal.rvs(
        mean=[10.0, 0.0],
        cov=Sigma,
        size=500
    )
    YP = np.ones(500)

    #stack everything together
    X = np.vstack([XQ, XP])
    y = np.concatenate([YQ, YP])

    #compute the KL divergence between the two distributions
    kl = kl_divergence_gaussians(
        mu1=np.array([10.0,0.0]),
        cov1=Sigma,
        mu2=np.array([0.0,0.0]),
        cov2=Sigma
        )

    #K means with no covariance modelling (Identity)
    k_mean_identity_labels, centers_identity, Sigmas_identity = k_means_clustering(X, n_clusters=2, covariance="identity", seed=1000+i)

    #K means with full covariance matrices
    k_means_full_labels, centers_full, Sigmas_full = k_means_clustering(X, n_clusters=2, covariance="full", seed=1000+i)

    #EM clustering.  Gaussian Mixture impliements EM algorithm.  
    #https://scikit-learn.org/stable/modules/mixture.html#gmm
    EM_cluster = GaussianMixture(n_components=2, n_init=1, random_state=1000+i)
    EM_cluster_labels = EM_cluster.fit_predict(X)

    #store what we need to plot the three clustering methods for the first four runs
    if i < 4:
        plot_runs.append({
            "run": i,
            "X": X,
            "y_true": y,
            "pred_id": k_mean_identity_labels,
            "pred_full": k_means_full_labels,
            "pred_em": EM_cluster_labels,
        })

    #adding everything to the results list
    results.append({
        "run": i,
        "KL_divergence": kl,
        "accuracy_kmeans_id": conting_matrix(y, k_mean_identity_labels),
        "ARI_kmeans_id": adjusted_rand_score(y, k_mean_identity_labels),
        "accuracy_kmeans_full": conting_matrix(y, k_means_full_labels),
        "ARI_kmeans_full": adjusted_rand_score(y, k_means_full_labels),
        "accuracy_EM": conting_matrix(y, EM_cluster_labels),
        "ARI_EM": adjusted_rand_score(y, EM_cluster_labels),
    })


#everything after this is just plotting stuff.  
#plotting the clustering results for the first four runs.
for item in plot_runs:

    run = item["run"]
    X = item["X"]

    #share the same axes for all 3 methods
    fig, axes = plt.subplots(1, 3, figsize=(15,5), sharex=True, sharey=True)

    #list out the titles, and labels for the three methods
    titles = ["K-means (Identity)", "K-means (Full)", "EM"]
    label_sets = [item["pred_id"], item["pred_full"], item["pred_em"]]

    #found this zip function.  Pretty nice.  Iterates through axes, labels, and titles.
    #https://www.codecademy.com/article/python-zip-function
    for ax, labels, title in zip(axes, label_sets, titles):

        for k in np.unique(labels):
            ax.scatter(X[labels==k,0], X[labels==k,1], s=10, alpha=0.7, label=f"Cluster {k}")

        ax.set_title(f"{title}\nRun {run}")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"assignments/hw08/figures/partb_run{run}_clusters.png", dpi=400, bbox_inches="tight")
    plt.close()


#plotting the accuracy bs KL divergence for the three methods across all 10 runs.
results_df = pd.DataFrame(results)

#add some jitter to see things better
jitter = 0.05
KL_id = results_df["KL_divergence"] + np.random.normal(0, jitter, size=len(results_df))
KL_full = results_df["KL_divergence"] + np.random.normal(0, jitter, size=len(results_df))
KL_em = results_df["KL_divergence"] + np.random.normal(0, jitter, size=len(results_df))

plt.figure()
plt.scatter(KL_id, results_df["accuracy_kmeans_id"], label="K-means Identity")
plt.scatter(KL_full, results_df["accuracy_kmeans_full"], label="K-means Full")
plt.scatter(KL_em, results_df["accuracy_EM"], label="EM")
plt.xlabel("KL Divergence")
plt.ylabel("Accuracy")
plt.title("Accuracy vs KL Divergence for 3 Clustering Methods\nSmall Jitter Added for Visualization")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("assignments/hw08/figures/KL_vs_Accuracy.png", dpi=400, bbox_inches="tight")
plt.close()
#plt.show()

#plotting the Adjusted Rand Index vs KL divergence for the three methods across all 10 runs
plt.figure()
plt.scatter(KL_id, results_df["ARI_kmeans_id"], label="K-means Identity")
plt.scatter(KL_full, results_df["ARI_kmeans_full"], label="K-means Full")
plt.scatter(KL_em, results_df["ARI_EM"], label="EM")
plt.xlabel("KL Divergence")
plt.ylabel("Adjusted Rand Index")
plt.title("Adjusted Rand Index vs KL Divergence for 3 Clustering Methods\nSmall Jitter Added for Visualization")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("assignments/hw08/figures/KL_vs_ARI.png", dpi=400, bbox_inches="tight")
plt.close()

#reporting the table and reordering the columns
df = results_df[[
"run",
"KL_divergence",
"accuracy_kmeans_id",
"ARI_kmeans_id",
"accuracy_kmeans_full",
"ARI_kmeans_full",
"accuracy_EM",
"ARI_EM"
]].round(3)

#rename the columns
df.columns = [
    "Run",
    "KL",
    "Acc KM (Id)",
    "ARI KM (Id)",
    "Acc KM (Full)",
    "ARI KM (Full)",
    "Acc EM",
    "ARI EM"
]

#need to send this table to LATEX
LATEX_table = df.to_latex(
    index=False,
    caption=None,
    label=None,
    column_format="c" * df.shape[1],
    escape=True,
    float_format="%.3f"
)

#save latex table
with open("assignments/hw08/output/final_table.tex", "w") as f:
    f.write(LATEX_table)