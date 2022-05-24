# %%
import gym
import d4rl
from utils.env import glance
import numpy as np

# %%
env = gym.make('halfcheetah-random-v0')
d1 = env.get_dataset()
d2 = d4rl.qlearning_dataset(env)
# it = d4rl.sequence_dataset(env, d4rl.qlearning_dataset(env))

# for i in it:
#   print(i)

print(d2)
# glance(env)
