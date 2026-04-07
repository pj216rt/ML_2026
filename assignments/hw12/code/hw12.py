import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random

#need to create a cartpole environment
env = gym.make("CartPole-v1")

n_episodes = 1000

#loop over and collect the states of 1000 episodes

for i in range(n_episodes):
    state, info = env.reset()

    print(f"Starting state: {state}")

    done = False

    while not done:
        a = random.randrange(2)
        st = env.step(a)
        next_state, reward, done = st[:3]

env.close()

#need to multiply by 10 and converting to integers