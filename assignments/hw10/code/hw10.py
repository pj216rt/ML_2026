import numpy as np
from scipy.sparse import diags
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image
from scipy.sparse import coo_matrix
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

                    #get RGB vector for neighbor
                    I_j = img_array[neighbor_row, neighbor_column]

                    #get neighbor index
                    j = (neighbor_row*width) + neighbor_column

                    #compute the squared eucliodian distance
                    diff = I_i - I_j
                    dist_sq = np.sum(diff**2)

                    #compute the affinity and store in the lists for the sparse matrix
                    affin = np.exp(-dist_sq / sigma**2)
                    rows.append(i)
                    cols.append(j)
                    data.append(affin)

    #build the sparse affinity matrix
    A = coo_matrix((data, (rows, cols)), shape=(n_pixels, n_pixels))

    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.tocsr.html
    return A.tocsr()

#function for spectral clustering
#build A on CPU.  Run svd_lowrank on torch
#lot of debugging stuff in here.  Had a hard time
def spectral_clustering(image_array, sigma, n_clusters, random_state):

    #image dimensions and total number of pixels
    height, width, channels = image_array.shape

    #set the torch device and random seed
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    #build the affinity matrix
    A = build_affinity_matrix(image_array, sigma)
    #print("build A")

    #need the Degree matrix.  Simply the sum of the four neighbors for each pixel
    #need sparse matrix
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.diags.html
    #flatten turns this into a simple 1D array
    D = np.array(A.sum(axis=1)).flatten()
    
    #need D inverse square root
    D_inv_sqrt = diags(D**(-0.5))

    #feed this into svd.  Pg 27 in notes
    L = D_inv_sqrt @ A @ D_inv_sqrt

    #convert to COO format for torch
    L_coo = L.tocoo()

    indices = np.vstack((L_coo.row, L_coo.col))
    indices = torch.from_numpy(indices).long().to(device)
    values = torch.from_numpy(L_coo.data).float().to(device)

    L_tensor = torch.sparse_coo_tensor(
        indices,
        values,
        size=L.shape,
        device=device
    )

    #run svd_lowrank on torch
    #https://pytorch.org/docs/stable/generated/torch.linalg.svd.html#torch.linalg.svd
    #q needs to be a "slightlyt overestimates rank of the matrix"
    #increaing niter can lead to better approximations
    U, S, Vh = torch.svd_lowrank(L_tensor, q=n_clusters+5, niter=10)
    #print("svd_lowrank") 

    #take first n_clusters columns of U
    #each row of E is the embedding of a pixel in the n_clusters dimensional space.  We will run k-means on these
    E = U[:, :n_clusters].cpu().numpy()

    #normalize the rows of E.  Need to avoid dividing by zero.  Ran into issues here if we didn't
    E_row_norms = np.linalg.norm(E, axis=1, keepdims=True)
    E_row_norms = np.maximum(E_row_norms, 1e-12)
    E_normalized = E / E_row_norms

    #k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(E_normalized)
    #print("kmeans")

    #reshape the labels back to the image dimensions for visualization
    label_image = labels.reshape(height, width)

    return label_image, labels


#actual problem
sigmas = [0.1, 0.2, 0.05]
clusters = [4,8]

#loop over the various sigma values
for sigma in sigmas:

    #loop over the clusters
    for cluster in clusters:
        
        #image dimensions and total number of pixels
        height, width, channels = img_array.shape

        #run clustering
        label_image, labels = spectral_clustering(
            #image_array=img_array,
            image_array=img_array,
            sigma=sigma,
            n_clusters=cluster, random_state=123
            )
        
        #plotting part a, clustering results
        #https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
        plt.figure()
        plt.imshow(label_image)
        plt.title(f"Spectral clustering with k = {cluster} clusters, sigma={sigma}")
        plt.axis("off")
        #plt.show()
        plt.savefig(f"assignments/hw10/figures/clusters_k{cluster}_sigma{sigma}.pdf", dpi=400, bbox_inches="tight")
        plt.close()

        #part b
        #for each cluster, compute the means of all R, G, and B values for the pixels in that cluster
        #place that mean at all locations of the pixels in that cluster.

        label_grid = labels.reshape(height, width)

        #don't need to specify dims with zeros_like
        color_image = np.zeros_like(img_array)

        #loop over each cluster
        for i in range(cluster):
            print(i)
            #find all pixels in cluster i
            indicator = (label_grid == i)

            #select pixels in the original image that belong to cluster i
            #shape should be (#pixels in cluster, 3)
            temp = img_array[indicator]

            #compute the mean R, G, and B values for the pixels in cluster i
            mean_red = temp[:, 0].mean()
            mean_green = temp[:, 1].mean()
            mean_blue = temp[:, 2].mean()
            mean_color = np.array([mean_red, mean_green, mean_blue])

            #replace all pixels in cluster with the mean color
            color_image[indicator] = mean_color
    
        #plot this mean color image
        plt.figure()
        plt.imshow(color_image)
        plt.title(f"Mean color image (k={cluster}, sigma={sigma})")
        plt.axis("off")
        #plt.show()
        plt.savefig(f"assignments/hw10/figures/mean_k{cluster}_sigma{sigma}.pdf", dpi=400, bbox_inches="tight")
        plt.close()
