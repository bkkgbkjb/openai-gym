import numpy as np
import setup
import torch
from algorithm import Simple, Preprocess
from utils.agent import Agent, OfflineAgent
import gym
from setup import RANDOM_SEED
from utils.env_sb3 import flat_to_episode, flat_to_transitions
from utils.reporter import get_reporter
from utils.env import make_envs, offline_train_and_eval, make_train_and_eval_env
from args import args, exp_name

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# %%
[env] = make_envs("hopper-medium-v2")

dataset = env.get_dataset()

last_done = (
    np.argwhere(np.logical_or(dataset["timeouts"], dataset["terminals"]) == True)[
        -1
    ].item()
    + 1
)

state_mean = np.mean(dataset["observations"][:last_done], axis=0)
state_std = np.std(dataset["observations"][:last_done], axis=0) + 1e-7

episodes = flat_to_episode(
    (dataset["observations"][:last_done] - state_mean) / state_std,
    dataset["actions"][:last_done],
    dataset["rewards"][:last_done].tolist(),
    np.logical_or(dataset["timeouts"][:last_done], dataset["terminals"][:last_done]),
    [
        dict(next_state=(ns - state_mean) / state_std)
        for ns in dataset["next_observations"][:last_done]
    ],
    True,
)


# %%

agent = OfflineAgent(
    Simple(env.observation_space.shape[0], env.action_space.shape[0]),
    Preprocess(),
)

agent.set_algm_reporter(get_reporter(agent.name, exp_name))

SINGLE_TRAIN: int = 128 * agent.algm.batch_size
offline_train_and_eval(
    agent,
    dict(episodes=episodes),
    env,
    single_train_frames=SINGLE_TRAIN,
    eval_per_train=1,
    total_train_frames=1000 * SINGLE_TRAIN,
)
