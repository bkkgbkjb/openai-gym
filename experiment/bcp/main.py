import setup
import torch
from algorithm import BC, Preprocess
from utils.agent import Agent, OfflineAgent
import gym
from setup import RANDOM_SEED
from utils.reporter import get_reporter
from utils.env import offline_train_and_eval, make_train_and_eval_env
from torch.utils.data import TensorDataset, DataLoader
from envs.LESSONs.goal_env.mujoco import *
import h5py

# %%
dataset = h5py.File('./dataset/AntMaze.hdf5', 'r')

tensor_data = TensorDataset(
    torch.from_numpy(dataset['state'][:]).type(torch.float32),
    torch.from_numpy(dataset['action'][:]).type(torch.float32),
    torch.from_numpy(dataset['reward'][:]).type(torch.float32),
    torch.from_numpy(dataset['next_state'][:]).type(torch.float32),
    torch.from_numpy(dataset['done'][:]).type(torch.float32),
    torch.from_numpy(dataset['info']['goal'][:]).type(torch.float32),
    )

dataloader = DataLoader(
    tensor_data,
    batch_size=256,
)
# %%

agent = OfflineAgent(
    BC(train_env.observation_space.shape[0], train_env.action_space.shape[0]),
    Preprocess())

agent.set_algm_reporter(get_reporter(agent.name))

offline_train_and_eval(agent,
                       dataloader,
                       eval_env,
                       single_train_frames=300 * 100,
                       eval_per_train=5,
                       total_train_frames=1000 * 300 * 100)

dataset.close()