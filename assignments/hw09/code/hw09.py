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

#get the result.  Also want to see the value of the functions at these minimized values
x_min, y_min = result.x
v1 = a((x_min*(sigma**2 + 1))/(2*sigma**2))
v2 = b(x_min, y_min)/2.0
v3 = a(y_min)

#put this into a dataframe for the writeup, rounded to 4 decimal places.
df1 = pd.DataFrame({
    "x_min": [x_min],
    "y_min": [y_min],
    "Expresssion 1": [v1],
    "Expression 2": [v2],
    "Expression 3": [v3]
}).round(4)

#need to send this table to LATEX
LATEX_table = df1.to_latex(
    index=False,
    caption=None,
    label=None,
    column_format="c" * df1.shape[1],
    escape=True
)

#save latex table
with open("assignments/hw09/output/smallest_test_error.tex", "w") as f:
    f.write(LATEX_table)

#plotting this.  Not needed, but a nice visual
x_vals = np.linspace(0.01,5,500)
y_vals = np.linspace(0.01,5,500)
X, Y = np.meshgrid(x_vals, y_vals)

vals = [X,Y]
Z = objective(vals)

plt.figure(figsize=(6,5))
plt.contourf(X,Y,Z,levels=50)
plt.scatter(x_min,y_min,color='black',s=80)
plt.colorbar(label="Distance between the three expressions")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Contour plot of distance between the three expressions")
plt.savefig("assignments/hw09/figures/contour_of_solution.pdf", dpi=400, bbox_inches="tight")
plt.close()


#problem 2, sequence of dice rolls x from a dishonest casino model.  
#states are fair and loaded.
#https://en.wikipedia.org/wiki/Viterbi_algorithm#Pseudocode
df_q2 = pd.read_csv("assignments/hw09/data/hmm_pb1.csv", header=None)

#states, 1 = Fair, 2 = Loaded
states = [1, 2]

#initial probabilities
pi = np.array([0.5, 0.5])

#need emission probabilities.  P(roll | state)
#fair die, 1/6 for each roll.  loaded die, 1/10 for 1-5 and 1/2 for 6.
emit = {
    1: {1: 1/6, 2: 1/6, 3: 1/6, 4: 1/6, 5: 1/6, 6: 1/6},   
    2: {1: 1/10, 2: 1/10, 3: 1/10, 4: 1/10, 5: 1/10, 6: 1/2}
}

#function for Viterbi algorithm
def Viterbi(states, init, trans, emit, obs):
    #get number of observations
    T = len(obs)
    print("Hello")