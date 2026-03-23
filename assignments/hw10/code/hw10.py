import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from scipy.sparse import diags
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from PIL import Image
from scipy.sparse import coo_matrix
import time
import torch

#found this for working with images
#https://pillow.readthedocs.io/en/stable/reference/Image.html
#https://stackoverflow.com/questions/51321960/import-image-in-python
#https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert

#load the image and convert to numpy array and normalize
img = Image.open('assignments/hw10/data/horse027.jpg').convert("RGB")
img_array = np.array(img)
img_array = img_array/255.0

#function for affinity matrix
#use sparse matrices for speed
#https://www.geeksforgeeks.org/python/how-to-create-a-sparse-matrix-in-python/
def build_affinity_matrix(img_array, sigma):
    height, width, channels = img_array.shape
    n_pixels = height*width
    rows = []
    cols = []
    data = []
    
    for row in range(height):
        for column in range(width):
            I_i = img_array[row, column]
            i = (row*width) + column

            for neighbor_row, neighbor_column in [(row-1, column), (row+1, column), (row, column-1), (row, column+1)]:
                if 0 <= neighbor_row < height and 0 <= neighbor_column < width:
                    I_j = img_array[neighbor_row, neighbor_column]
                    j = (neighbor_row*width) + neighbor_column

                    diff = I_i - I_j
                    dist_sq = np.sum(diff**2)

                    affin = np.exp(-dist_sq / sigma**2)
                    rows.append(i)
                    cols.append(j)
                    data.append(affin)
    A = coo_matrix((data, (rows, cols)), shape=(n_pixels, n_pixels))

    return A.tocsr()

#function for spectral clustering
#build A on CPU.  Run svd_lowrank on torch
def spectral_clustering(image_array, sigma, n_clusters, random_state):
    height, width, channels = image_array.shape
    n_pixels = height*width

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(random_state)

    #build the affinity matrix
    t0 = time.time()
    A = build_affinity_matrix(image_array, sigma)
    t1 = time.time()
    print(f"build A: {t1 - t0:.2f} sec")

    #need the Degree matrix.  Simply the sum of the four neighbors for each pixel
    #need sparse matrix
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.diags.html
    D = np.array(A.sum(axis=1)).flatten()
    D = np.maximum(D, 1e-10)  #avoid dividing by 0
    print("min degree:", D.min(), "max degree:", D.max(), "num zeros:", np.sum(D == 0))
    
    #need D inverse square root
    D_inv_sqrt = diags(D**(-0.5))

    #issue here for the last loop through


    #feed this into svd.  Pg 27 in notes
    L = D_inv_sqrt @ A @ D_inv_sqrt
    L = L.tocoo()
    t2 = time.time()
    print(f"L: {t2 - t1:.2f} sec")

    #need to convert the matrix to a sparse tensor for torch
    indices = np.vstack((L.row, L.col))
    indices = torch.from_numpy(indices).long()
    values = torch.from_numpy(L.data).float()

    L_tensor = torch.sparse_coo_tensor(
        indices,
        values,
        size=L.shape,
        device=device
    ).coalesce()

    #run svd_lowrank on torch
    #https://pytorch.org/docs/stable/generated/torch.linalg.svd.html#torch.linalg.svd
    U, S, Vh = torch.svd_lowrank(L_tensor, q=n_clusters, niter=5)
    t3 = time.time()
    print(f"svd_lowrank: {t3 - t2:.2f} sec") 

    #move U back to cpu
    U = U.cpu().numpy()
    print("U finite before norm:", np.all(np.isfinite(U)))

    #normalize the rows to have unit length
    U_rows_norm = np.linalg.norm(U, axis=1, keepdims=True)
    print("min row norm:", U_rows_norm.min(), "num zero row norms:", np.sum(U_rows_norm <= 1e-12))
    U_rows_norm = np.maximum(U_rows_norm, 1e-12)
    U_normalized = U / U_rows_norm
    print("normalized")

    #run k means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(U_normalized)
    t4 = time.time()
    print(f"kmeans: {t4 - t3:.2f} sec")

    label_image = labels.reshape(height, width)

    return label_image, labels, A

sigmas = [0.1, 0.2, 0.05]

#working with a smaller version of the image
# img_small = img_array[::2, ::2, :]

for sigma in sigmas:
    label_image, labels, A = spectral_clustering(
        #image_array=img_array,
        image_array=img_array,
        sigma=sigma,
        n_clusters=4, random_state=123
    )

    #plotting part a
    plt.figure()
    plt.imshow(label_image, cmap="tab10")
    plt.title(f"Spectral clustering with 4 clusters, sigma={sigma}")
    plt.axis("off")
    plt.show()

    #part b
    #getting the mean of all RGB values
    