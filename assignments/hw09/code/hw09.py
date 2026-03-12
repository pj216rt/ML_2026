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

#part a, Virterbi algo.

#states, 1 = Fair, 2 = Loaded
states = [1, 2]

#initial probabilities
pi = np.array([0.5, 0.5])

#chose these probabilties from
#https://hmmlearn.readthedocs.io/en/latest/auto_examples/plot_casino.html
trans = np.array([
    [0.95, 0.05],   #fair
    [0.10, 0.90]    #loaded
])

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
    S = len(states)

    #need log probabiilities
    log_init = np.log(init)
    log_trans = np.log(trans)

    #need prob and prev.  need prev to be integer, bc it is storing the state
    #init as -ingf bc of log space
    log_prob = np.full((T, S), -np.inf)
    prev = np.zeros((T, S), dtype=int)

    #initialize 
    #need to remember that python indexes start from 0, need to subtract 1 from state to get the right index in the prob array
    for s in states:
        log_prob[0][s-1] = log_init[s-1] + np.log(emit[s][obs[0]])

    #recursion portion
    for t in range(1, T):
        for s in states:
            for r in states:
                #summations bc of log space
                new_prob = (
                    log_prob[t-1][r-1]
                    + log_trans[r-1][s-1]
                    + np.log(emit[s][obs[t]])
                )

                if new_prob > log_prob[t][s-1]:
                    log_prob[t][s-1] = new_prob
                    prev[t][s-1] = r
    
    #empty array of length T.  Need this to be integer
    path = np.zeros(T, dtype=int)
    path[T - 1] = np.argmax(log_prob[T - 1]) + 1

    for t in range(T - 2, -1, -1):
        path[t] = prev[t + 1][path[t+1] - 1]

    return path

path = Viterbi(states=states, init=pi, trans=trans, emit=emit, obs=df_q2.iloc[0].to_numpy())

#need to do something with this.  Find a beter way to report

#implement the forward algorithm for the same sequence of dice rolls.
#need to renormalize the alpha values at each step to prevent underflow.
#for forward pass, store scaling factor at each time step
def forward_algorithm(states, init, trans, emit, obs):
    #length of the rice rolls
    T = len(obs)

    #number of hidden states
    S = len(states)

    #initialize alpha.  alpha[t,s] is the forward probability \alpha_t^s.
    #each row is a time t and each column is a state s
    alpha = np.zeros((T, S))

    #scaling factor used at time t.  
    u = np.zeros(T)

    #\alpha_1^k = \pi_k * b_{x_1}^k
    for s in states:
        alpha[0][s-1] = init[s-1] * emit[s][obs[0]]
    
    #scaling alpha 0
    u[0] = np.sum(alpha[0])
    alpha[0] = alpha[0] / u[0]

    for t in range(1, T):
        for s in states:
            sum_alpha = 0

            #sum over all possible previous states r.
            for r in states:
                sum_alpha += alpha[t-1][r-1] * trans[r-1][s-1]
            
            #multiply by the emission prob
            alpha[t][s-1] = sum_alpha * emit[s][obs[t]]
        
        #scaling alpha t
        u[t] = np.sum(alpha[t])
        alpha[t] = alpha[t] / u[t]
    
    #asked to report the ratio of the alpha values at time 118.
    #time 118 -> row 118
    #state 1 is column 0, state 2 is column 1.  So we want alpha[118,0] / alpha[118,1]
    ratio = alpha[118,0] / alpha[118,1]

    #ratio, alpha, and u.  u is needed for backward algo
    return ratio, u

#function to implement the backward algorithm for the same sequence of dice rolls.
#using the algorithm from pg 23 of the slide
def backward_algorithm(states, trans, emit, obs, u):
    #number of observations
    T = len(obs)

    #number of hidden states
    S = len(states)

    #store the backward probabilities.  beta[t,s] is the backward probability \beta_t^s.
    beta = np.zeros((T, S))

    #initialize beta.  \beta_T^k = 1 for all k.
    beta[T-1] = np.ones(S)

    for t in range(T-2, -1, -1):
        for s in states:
            sum_beta = 0

            #sum over all possible next states r.
            for r in states:
                sum_beta += trans[s-1][r-1] * emit[r][obs[t+1]] * beta[t+1][r-1]

            beta[t][s-1] = sum_beta/u[t+1]
    
    #asked to report the ratio of the alpha values at time 118.
    #time 118 -> row 118
    #state 1 is column 0, state 2 is column 1.  So we want beta[118,0] / beta[118,1]
    ratio = beta[118,0] / beta[118,1]
    return ratio

#Question 3, Implement and run the Baum-Welch algorithm using the Forward and Backward
#algorithms from q2.  Initialize \pi, a, b with a guess of our choise

alpha_ratio, u = forward_algorithm(states=states, init=pi, trans=trans, emit=emit, obs=df_q2.iloc[0].to_numpy())
beta_ratio = backward_algorithm(states=states, trans=trans, emit=emit, obs=df_q2.iloc[0].to_numpy(), u=u)

#Question 3, impliment the Baum-Welch algorithm using the forward and backward algorithms from q2.  
#Initialize \pi, a, b with a guess of our choice.  report the final values of \pi, a, b.

df_q3 = pd.read_csv("assignments/hw09/data/hmm_pb2.csv", header=None)
print(df_q3.head())

#need to define the Baum-Welch algorithm.
def baum_welch(seed=123):
    #define the random generator for reproducibility
    rng = np.random.default_rng(seed)
    
    #need to initialize parameters at random.
    pi = 
    pass
