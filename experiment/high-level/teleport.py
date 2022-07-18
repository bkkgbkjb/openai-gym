import gym
import torch
import numpy as np

class Teleport(gym.Wrapper):
	def __init__(self, env: gym.Env, target: np.ndarray):
		super().__init__(env)
		self.target = target
	
	def step(self, action):
		if (action[2:] == self.target).all():
			self.env.unwrapped._wrapped_env.set_xy((action[0], action[1]))
		return self.env.step(action)
