import setup
import torch
from algorithm import CQL_SAC, Preprocess
from utils.agent import Agent, OfflineAgent
import gym
from setup import RANDOM_SEED
from utils.reporter import get_reporter
from utils.env import offline_train_and_eval, make_train_and_eval_env
from utils.env_sb3 import flat_to_transitions
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# %%
train_env, eval_env = make_train_and_eval_env("walker2d-expert-v2", [],
                                              RANDOM_SEED)

dataset = train_env.get_dataset()

transitions = flat_to_transitions(
    dataset["observations"],
    dataset["actions"],
    dataset["rewards"],
    np.logical_or(dataset["timeouts"], dataset["terminals"]),
)

# %%

agent = OfflineAgent(
    CQL_SAC(train_env.observation_space.shape[0],
            train_env.action_space.shape[0]), Preprocess())

agent.set_algm_reporter(get_reporter(agent.name))

offline_train_and_eval(agent,
                       dict(transitions=transitions),
                       eval_env,
                       eval_per_train=5,
                       single_train_frames=300 * 256,
                       total_train_frames=1000 * 300 * 256)
