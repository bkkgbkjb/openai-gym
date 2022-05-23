import setup
from algorithm import TD3, Preprocess
from utils.agent import Agent
import gym
from utils.env import train, train_and_eval, make_train_and_eval_env
from setup import RANDOM_SEED

from utils.reporter import get_reporter

# %%

train_env, eval_env = make_train_and_eval_env("Walker2d-v2", [], RANDOM_SEED)

# %%

agent = Agent(
    train_env,
    TD3(train_env.observation_space.shape[0], train_env.action_space.shape[0]),
    Preprocess())

agent.set_algm_reporter(get_reporter(agent.name))

train_and_eval(agent, eval_env)
