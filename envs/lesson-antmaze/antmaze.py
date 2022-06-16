import setup
import numpy as np
import gym
import torch
from setup import RANDOM_SEED

from goal_env.mujoco import *

env = gym.make('AntMaze1-v1')
env.seed(RANDOM_SEED)