import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#question 1.  
sigma = 0.5

#define the functions 
def a(x):
    temp  = x - 1 - np.log(x)
    return temp

def b(x,y):
    temp = (1+y-x)**2/(1+(2*y))
    return temp

#"as close to each other as possible".  Objective function?
def objective(vars):
    x, y = vars

    variable1 = a((x*(sigma**2 + 1))/(2*sigma**2))
    variable2 = b(x, y)/2.0
    variable3 = a(y)

    #Euclidean distance between the three variables.  Want to minimize this.
    #pdist wasn't working, so I just wrote it out.
    temp = ((variable1 - variable2)**2 +
            (variable1 - variable3)**2 +
            (variable2 - variable3)**2)

    return temp

#minimize this all.  Need an itial guess.  Let's say x = 1, y = 1
#need bounds though, because log(x) is undefined for x <= 0.  denominator in b(x,y) just can't be -0.5, which is covered if
#we bound both x and y > 0.  bounds give a minimum and no maximum.
initial_guess = [1, 1]
bounds = [(1e-8, None), (1e-8, None)]
result = minimize(objective, initial_guess, bounds=bounds)

#get the result
x_min, y_min = result.x
