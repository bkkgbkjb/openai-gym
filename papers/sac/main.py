import setup
from time import sleep, time
from algorithm import SAC, Preprocess
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

# %%
RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# %%
writer = SummaryWriter()

# %%


def glance():
    env = gym.make("Walker2d-v2")
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)
    env.observation_space.seed(RANDOM_SEED)
    env.reset()

    print(env.action_space, env.observation_space)
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


def train():
    env = gym.make("Walker2d-v2")
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)
    env.observation_space.seed(RANDOM_SEED)
    env.reset()

    TRAINING_TIMES = 1e6

    print(env.action_space, env.observation_space)
    agent = Agent(env, SAC(17, 6), Preprocess())
    agent.reset()
    print(f"train agent: {agent.name}")

    j = 0

    def report(info: Dict[Any, Any]):
        nonlocal j
        # writer.add_scalar("target", info['target'], j)
        # writer.add_scalar("policy_loss", info['policy_loss'], j)
        # writer.add_scalar("entropy", info['entropy'], j)
        # writer.add_scalar("value_loss", info['value_loss'], j)
        for k, v in info.items():
            writer.add_scalar(k, v, j)
        j += 1

    agent.set_algm_reporter(report)

    with tqdm(total=TRAINING_TIMES) as pbar:
        frames = 0
        epi = 0
        while frames < TRAINING_TIMES:
            agent.reset()
            i = 0
            end = False
            while not end and frames < TRAINING_TIMES:

                (_, end) = agent.step()
                i += 1

            epi += 1
            frames += i

            pbar.update(i)

            rwd = np.sum([r for r in agent.reward_episode])
            writer.add_scalar("reward", rwd, epi)


# %%
train()
