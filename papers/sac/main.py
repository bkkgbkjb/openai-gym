import setup
from time import sleep, time
from algorithm import RandomAlgorithm, Preprocess
import torch
from os import times
import plotly.graph_objects as go
from tqdm import tqdm
from utils.agent import Agent
from typing import Dict, List, Any
import gym
from utils.env import PreprocessObservation, FrameStack, ToTensorEnv
from utils.env_sb3 import WarpFrame, MaxAndSkipEnv, NoopResetEnv, EpisodicLifeEnv
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

# %%
RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# %%
env = gym.make("Walker2d-v2")
env.seed(RANDOM_SEED)
env.action_space.seed(RANDOM_SEED)
env.observation_space.seed(RANDOM_SEED)
env.reset()

print(env.action_space, env.observation_space)

writer = SummaryWriter()


# %%
for _ in range(3):
    r = 0.0
    env.reset()
    s = False
    t = 0

    while not s:
        env.render(mode='human')
        sleep(0.05)

        (_, rwd, stop, _) = env.step(env.action_space.sample())
        t += 1

        r += rwd

        if stop:
            print(f'rwd is: {r}, steps: {t}')
            break

# %%
