import setup
import gym
from environments.create_maze_env import create_maze_env
from utils.env import glance

env = create_maze_env('AntMaze')
glance(env, 'human')