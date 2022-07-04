import setup
import torch
from algorithm import BCP, Preprocess
from utils.agent import Agent, OfflineAgent
import gym
from setup import RANDOM_SEED
from utils.reporter import get_reporter
from utils.env import offline_train_and_eval, make_train_and_eval_env
from torch.utils.data import TensorDataset, DataLoader
from goal_env.mujoco import *
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

test_env = gym.make('AntMaze1Test-v1')

agent = OfflineAgent(
    BCP(test_env.observation_space['observation'].shape[0],
        test_env.observation_space['desired_goal'].shape[0],
        test_env.action_space.shape[0], 16), Preprocess())

agent.set_algm_reporter(get_reporter(agent.name))

offline_train_and_eval(agent,
                       dataloader,
                       test_env,
                       single_train_frames=300 * 100,
                       eval_per_train=5,
                       total_train_frames=1000 * 300 * 100)

dataset.close()