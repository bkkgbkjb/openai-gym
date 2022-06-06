# %%
import setup
from setup import RANDOM_SEED
from algorithm import PPO, Preprocess
from utils.agent import Agent
import gym
from utils.env import make_train_and_eval_env, train_and_eval
from utils.env_sb3 import SkipFrames, FrameStack, PreprocessObservation
from utils.reporter import get_reporter

# %%
wrappers = [
    lambda env: SkipFrames(env), lambda env: PreprocessObservation(env),
    lambda env: FrameStack(env, num_stack=4)
]
train_env, eval_env = make_train_and_eval_env("PongDeterministic-v4", wrappers,
                                              RANDOM_SEED)

# %%

agent = Agent(train_env, PPO(train_env.action_space.n), Preprocess())

agent.set_algm_reporter(get_reporter(agent.name))

train_and_eval(agent, eval_env, total_train_frames=int(1e6))
