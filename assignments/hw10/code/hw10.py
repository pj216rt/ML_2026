import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

#found this for working with images
#https://pillow.readthedocs.io/en/stable/reference/Image.html
#https://stackoverflow.com/questions/51321960/import-image-in-python
#https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert

#load the images
img = Image.open('assignments/hw10/data/horse027.jpg').convert("RGB")

#convert to numpy array
img_array = np.array(img)

#normalize image
img_array = img_array / 255.0

#get dims of img_array
height, width, channels = img_array.shape
n_pixels = height*width

sigma = 0.1

#initialize an empty affinity matrix
A = np.zeros((n_pixels, n_pixels))

#affinity matrix.  For each pixel, get its RGB value, and compare its four nieghbors.  
#loop over every pixel.  Need to convert from row,column to a single index
for row in range(height):
    for column in range(width):
        #get RGB triplet
        I_i = img_array[row, column]

        #loop over the four neighbors
        for neighbor_row, neighbor_column in [(row-1, column), (row+1, column), (row, column-1), (row, column+1)]:
            #check if neighbor is within bounds
            if 0 <= neighbor_row < height and 0 <= neighbor_column < width:
                #get RGB triplet of neighbor
                I_j = img_array[neighbor_row, neighbor_column]

                diff = I_i - I_j
                dist_sq = np.sum(diff**2)

                #compute affinity using the formula:
                A[row, column, neighbor_row, neighbor_column] = np.exp(-dist_sq / sigma**2)