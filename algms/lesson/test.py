import setup
from papers.lesson.lesson import LESSON, Preprocess
from utils.agent import Agent
import gym
from setup import RANDOM_SEED
from utils.reporter import get_reporter
from utils.env import train_and_eval, make_train_and_eval_env
from goal_env.mujoco import *
from gym.wrappers import rescale_action

# %%
train_env, eval_env = make_train_and_eval_env("AntMaze1-v1", [], RANDOM_SEED)
# %%
train_env._max_episode_steps = 3000

for i in range(4000):
	(_, _ , s, _) = train_env.step(train_env.action_space.sample())
	if s:
		print(f'end: {i}')
		break