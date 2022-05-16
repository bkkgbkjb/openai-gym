import setup
from setup import RANDOM_SEED
from algorithm import DDPG, Observation, Preprocess
import plotly.graph_objects as go
from tqdm import tqdm
from utils.agent import Agent
from typing import Dict, List, Any
import gym
from gym.wrappers import RecordVideo
from algorithm import State, Observation, Action
from utils.env import PreprocessObservation, FrameStack, ToTensorEnv
from utils.env_sb3 import WarpFrame, MaxAndSkipEnv, NoopResetEnv, EpisodicLifeEnv
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# %%
writer = SummaryWriter()


# %%
def train() -> Agent[Observation, State, Action]:
    env = gym.make("Walker2d-v2")
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)
    env.observation_space.seed(RANDOM_SEED)
    # env = RecordVideo(env, "rl-video")
    env.reset()

    TRAINING_TIMES = int(1e6)

    print(env.action_space, env.observation_space)
    agent = Agent(env, DDPG(17, 6), Preprocess())
    agent.reset()
    print(f"train agent: {agent.name}")

    j = 0

    def report(info: Dict[str, Any]):
        nonlocal j
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
        print(f"training end after {frames} frames")

    return agent


# %%
train()
