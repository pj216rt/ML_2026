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

sigma = 0.1

#affinity matrix.  For each pixel, get its RGB value, and compare its four nieghbors.  
#loop over every pixel
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

                #compute affinity using the formula:
                affinity = np.exp(-np.linalg.norm(I_i - I_j)**2 / (2*sigma**2))