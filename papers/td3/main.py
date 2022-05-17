import setup
from algorithm import TD3, Preprocess
from utils.agent import Agent
import gym
from utils.env import train

from utils.reporter import get_reporter

# %%

env = gym.make("Walker2d-v2")

agent = Agent(
    env, TD3(env.observation_space.shape[0], env.action_space.shape[0]), Preprocess()
)


agent.set_algm_reporter(get_reporter())

# %%
train(env, agent)
