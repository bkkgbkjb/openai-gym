import setup
import gym
import mujoco_maze
from utils.env import glance

env = gym.make('AntUMaze-v0')
glance(env, 'human')
