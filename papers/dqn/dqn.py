# %%
import setup
from algorithm import DQNAlgorithm, Preprocess
from utils.agent import Agent
import gym
# from gym.wrappers.frame_stack import FrameStack
from utils.env import PreprocessObservation, train_and_eval
from utils.env_sb3 import SkipFrames, FrameStack
from utils.reporter import get_reporter

# %%
env = gym.make("PongDeterministic-v4")
env.seed()
env.reset()

# %%
env = SkipFrames(env)
env = PreprocessObservation(env)
env = FrameStack(env, num_stack=4)

# %%

agent = Agent(env, DQNAlgorithm(env.action_space.n), Preprocess())

agent.set_algm_reporter(get_reporter())

train_and_eval(agent)
