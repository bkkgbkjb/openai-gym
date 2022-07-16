# %%
import setup
from setup import RANDOM_SEED
from algorithm import DQNAlgorithm, Preprocess
from utils.agent import Agent
import gym
# from gym.wrappers.frame_stack import FrameStack
from utils.env import make_train_and_eval_env, train_and_eval
from utils.env_sb3 import SkipFrames, FrameStack, PreprocessObservation
from utils.reporter import get_reporter

# %%
# eval_env = SkipFrames(eval_env)
# eval_env = PreprocessObservation(eval_env)
# eval_env = FrameStack(eval_env, num_stack=4)
wrappers = [
    lambda env: SkipFrames(env), lambda env: PreprocessObservation(env),
    lambda env: FrameStack(env, num_stack=4)
]
train_env, eval_env = make_train_and_eval_env("PongDeterministic-v4", wrappers,
                                              RANDOM_SEED)

# %%

agent = Agent(DQNAlgorithm(train_env.action_space.n), Preprocess())

agent.set_algm_reporter(get_reporter(agent.name))

train_and_eval(agent, train_env, eval_env, total_train_frames=int(3e6))
