import setup
from algorithm import DDPG, Preprocess
from setup import RANDOM_SEED
from utils.reporter import get_reporter
from utils.agent import Agent
import gym
from algorithm import Observation
from utils.env import train_and_eval
from utils.env import make_train_and_eval_env, train_and_eval

# %%
train_env, eval_env = make_train_and_eval_env("Walker2d-v2", [], RANDOM_SEED)

# %%

agent = Agent(
    DDPG(train_env.observation_space.shape[0], train_env.action_space.shape[0],
         1), Preprocess())

agent.set_algm_reporter(get_reporter(agent.name))

train_and_eval(agent, train_env, eval_env)
