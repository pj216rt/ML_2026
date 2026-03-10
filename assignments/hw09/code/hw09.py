import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#question 1.  
sigma = 0.5

#define the functions 
def a(x):
    temp  = x - 1 - np.log(x)
    return temp

def b(x,y):
    temp = (1+y-x)**2/(1+(2*y))
    return temp
