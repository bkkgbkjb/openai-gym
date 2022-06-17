import gym
from goal_env.mujoco import *
import numpy as np
import torch

env = gym.make('AntMaze1-v1')


env.reset()
for i in range(600):
	(_, _, done, _) = env.step(env.action_space.sample())

	if done:
		print(f'done at {i}')
