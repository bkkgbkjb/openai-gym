import setup
from sac import NewSAC, Preprocess
from utils.agent import Agent
import gym
from setup import RANDOM_SEED
from utils.reporter import get_reporter
from utils.env import train_and_eval, make_train_and_eval_env

# %%
train_env, eval_env = make_train_and_eval_env("Walker2d-v2", [], RANDOM_SEED)

# %%

agent = Agent(
    train_env,
    NewSAC(train_env.observation_space.shape[0],
           train_env.action_space.shape[0], 1, False, False), Preprocess())

agent.set_algm_reporter(get_reporter(agent.name))

train_and_eval(agent, eval_env)
