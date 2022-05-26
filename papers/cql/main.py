import setup
import torch
from algorithm import OfflineSAC, Preprocess
from utils.agent import Agent, OfflineAgent
import gym
from setup import RANDOM_SEED
from utils.reporter import get_reporter
from utils.env import offline_train_and_eval, make_train_and_eval_env
from torch.utils.data import TensorDataset, DataLoader

# %%
train_env, eval_env = make_train_and_eval_env("walker2d-medium-v2", [],
                                              RANDOM_SEED)

dataset = train_env.get_dataset()

tensor_data = TensorDataset(
    torch.from_numpy(dataset['observations']).type(torch.float32),
    torch.from_numpy(dataset['actions']).type(torch.float32),
    torch.from_numpy(dataset['rewards']).type(torch.float32),
    torch.from_numpy(dataset['next_observations']).type(torch.float32),
    torch.from_numpy(dataset['terminals']).type(torch.float32))

dataloader = DataLoader(
    tensor_data,
    batch_size=256,
)
# %%

agent = OfflineAgent(
    dataloader,
    OfflineSAC(train_env.observation_space.shape[0],
               train_env.action_space.shape[0]), Preprocess())

agent.set_algm_reporter(get_reporter(agent.name))

offline_train_and_eval(agent,
                       eval_env,
                       single_train_frames=256 * 10,
                       total_train_frames=dataset['observations'].shape[0])
