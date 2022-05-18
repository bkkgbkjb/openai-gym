import setup
from algorithm import TD3, Preprocess
from utils.agent import Agent
import gym
from utils.env import train, train_and_eval
from setup import RANDOM_SEED

from utils.reporter import get_reporter

# %%

env = gym.make("Walker2d-v2")
env.seed(RANDOM_SEED)
env.action_space.seed(RANDOM_SEED)
env.observation_space.seed(RANDOM_SEED)
env.reset()

agent = Agent(
    env, TD3(env.observation_space.shape[0], env.action_space.shape[0]), Preprocess()
)


agent.set_algm_reporter(get_reporter(agent.name))

# %%
train_and_eval(agent)
