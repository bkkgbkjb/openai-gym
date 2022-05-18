import setup
from algorithm import DDPG, Preprocess
from setup import RANDOM_SEED
from utils.reporter import get_reporter
from utils.agent import Agent
import gym
from algorithm import Observation
from utils.env import train_and_eval

# %%

env = gym.make("Walker2d-v2")
env.seed(RANDOM_SEED)
env.action_space.seed(RANDOM_SEED)
env.observation_space.seed(RANDOM_SEED)
env.reset()

agent = Agent(
    env, DDPG(env.observation_space.shape[0], env.action_space.shape[0]), Preprocess()
)


agent.set_algm_reporter(get_reporter())

train_and_eval(agent)
