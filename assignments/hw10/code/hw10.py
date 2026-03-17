import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from scipy.sparse import diags
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image
from scipy.sparse import coo_matrix

#found this for working with images
#https://pillow.readthedocs.io/en/stable/reference/Image.html
#https://stackoverflow.com/questions/51321960/import-image-in-python
#https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert

#load the image and convert to numpy array and normalize
img = Image.open('assignments/hw10/data/horse027.jpg').convert("RGB")
img_array = np.array(img)
img_array = img_array / 255.0

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
def spectral_clustering(image_array, sigma, n_clusters, random_state):
    height, width, channels = image_array.shape
    n_pixels = height*width

    #build the affinity matrix
    A = build_affinity_matrix(image_array, sigma)
    print(A.shape)

    #need the Degree matrix.  Simply the sum of the four neighbors for each pixel
    #need sparse matrix
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.diags.html
    D = np.array(A.sum(axis=1)).flatten()
    
    #need D inverse square root
    D_inv_sqrt = diags(D**(-0.5))
    print(D_inv_sqrt.shape)

    #feed this into svd.  Pg 27 in notes
    D_mangled = D_inv_sqrt @ A @ D_inv_sqrt

    U, S, VT = svds(D_mangled, k=n_clusters)

    #normalize the rows to have unit length
    U_rows_norm = np.linalg.norm(U, axis=1, keepdims=True)
    U_normalized = U / U_rows_norm

    #run k means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(U_normalized)

    label_image = labels.reshape(height, width)

    return label_image, labels, A

#part a
sigmas = [0.1, 0.2, 0.05]

for sigma in sigmas:
    label_image, labels, A = spectral_clustering(
        image_array=img_array,
        sigma=sigma,
        n_clusters=4, random_state=123
    )