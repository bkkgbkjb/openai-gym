import setup
from algorithm import LESSON, Preprocess
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

agent = Agent(
    train_env,
    LESSON(train_env.observation_space.shape[0],
           train_env.action_space.shape[0], train_env.goal_space[0]),
    Preprocess())

agent.set_algm_reporter(get_reporter(agent.name))

train_and_eval(agent, eval_env)
