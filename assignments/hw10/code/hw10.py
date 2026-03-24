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
    #get image dimensions and total number of pixels
    height, width, channels = img_array.shape
    n_pixels = height*width

    #initialize lists to build sparse matrix
    rows = []
    cols = []
    data = []
    
    #loop over each pixel in the image
    for row in range(height):
        for column in range(width):

            #get the RGB vector for the current pixel
            I_i = img_array[row, column]

            #get index of pixel.  each row has width elements, and before row r, there r width elements.  then add the column index
            i = (row*width) + column

            #loop over the neighbors (up, down, left, right) and compute the affinity.  Need to check for boundary conditions
            for neighbor_row, neighbor_column in [(row-1, column), (row+1, column), (row, column-1), (row, column+1)]:

                #check if the neighbor is within the image boundaries
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

    #feed this into svd.  Pg 27 in notes
    L = D_inv_sqrt @ A @ D_inv_sqrt
    L = L.tocoo()
    t2 = time.time()
    print(f"L: {t2 - t1:.2f} sec")

    #need to convert the matrix to a sparse tensor for torch.
    #the linalg>svds just took too long and didn't converge
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
    U, S, Vh = torch.svd_lowrank(L_tensor, q=n_clusters+5, niter=10)
    t3 = time.time()
    print(f"svd_lowrank: {t3 - t2:.2f} sec") 

    #low rank approximation.  Take the first n_clusters columns of U
    E = U[:, :n_clusters].cpu().numpy()
    print("E finite before norm:", np.all(np.isfinite(E)))

    #normalize rows
    E_row_norms = np.linalg.norm(E, axis=1, keepdims=True)
    print("min row norm:", E_row_norms.min(), "num zero row norms:", np.sum(E_row_norms <= 1e-12))
    E_row_norms = np.maximum(E_row_norms, 1e-12)
    E_normalized = E / E_row_norms

    #kmeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(E_normalized)
    t4 = time.time()
    print(f"kmeans: {t4 - t3:.2f} sec")

    #debug
    print("cluster sizes:", np.bincount(labels))

    label_image = labels.reshape(height, width)

    return label_image, labels, A

sigmas = [0.1, 0.2, 0.05]

#working with a smaller version of the image
# img_small = img_array[::2, ::2, :]

for sigma in sigmas:
    height, width, channels = img_array.shape
    n_pixels = height*width

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
    #getting the mean of all RGB values.  Need the actual RGB values

    #initialize empty pixel map
    mean_pixels = np.zeros_like(img_array)

    label_grid = labels.reshape(height, width)
    color_image = np.zeros_like(img_array)
    
    for i in range(4):
        mask = (label_grid == i)
        if np.any(mask):
            color_image[mask] = img_array[mask].mean(axis=0)
    
    #color_image = mean_pixels.reshape(height, width, 3)
    plt.figure()
    plt.imshow(color_image)
    plt.title(f"Mean-color image (Sigma={sigma})")
    plt.axis("off")
    plt.show()
    