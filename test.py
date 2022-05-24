# %%
import gym
import d4rl
from utils.env import glance
import numpy as np

# %%
env = gym.make('halfcheetah-random-v0')
dataset1 = env.get_dataset()
dataset2 = d4rl.qlearning_dataset(env)
# it = d4rl.sequence_dataset(env, d4rl.qlearning_dataset(env))

# for i in it:
#   print(i)

print(dataset2)
# glance(env)
