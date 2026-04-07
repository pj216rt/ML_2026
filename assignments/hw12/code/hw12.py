import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random

#need to create a cartpole environment
env = gym.make("CartPole-v1")

n_episodes = 1000

states = []

#part a 
#loop over and collect the states of 1000 episodes
for i in range(n_episodes):
    state, info = env.reset()

    done = False

    while not done:
        #append to the states list
        states.append(state)

        a = random.randrange(2)
        observation, reward, terminated, truncated, info = env.step(a)
        
        done = terminated or truncated

        #update state
        state = observation

env.close()

#need to multiply by 10 and converting to integers
states = (np.array(states)*10).astype(int)

#plot histograms and save
for i in range(4):
    plt.figure()
    plt.hist(states[:, i], bins=50)
    plt.title(f"Histogram of state variable {i}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(f"assignments/hw12/figures/hist_state_variable_{i}.pdf", dpi=400, bbox_inches="tight")
    plt.close()
    #plt.show()

#need to find the range of each of the state variables
#also need to report the number of discrete sate values for each state variable
mins = states.min(axis=0)
maxs = states.max(axis=0)
num_vals = maxs - mins + 1

#get total number of discrete states
num_total_states = np.prod(num_vals)
print(f"Total number of discrete states: {num_total_states}")

for i in range(4):
    print(f"State variable {i}: min={mins[i]}, max={maxs[i]}, num discrete values={num_vals[i]}")


#part b.  Need to use the state space to initialze Q table.  