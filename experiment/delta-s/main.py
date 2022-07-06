import setup
import torch
from algorithm import DeltaS, Preprocess
from utils.agent import Agent, OfflineAgent
import gym
from setup import RANDOM_SEED
from utils.reporter import get_reporter
from utils.env import make_envs, offline_train_and_eval, make_train_and_eval_env, record_video
from torch.utils.data import TensorDataset, DataLoader
from goal_env.mujoco import *
from arguments.arguments_hier_sac import args
import h5py

# %%
dataset = h5py.File(f'./dataset/{args.dataset}', 'r')

# %%

# test_env1 = gym.make('AntMaze1Test-v1')
test_env1 = gym.make(args.test)
test_env1.env_info = dict(goal_type='farthest')
# test_env2 = gym.make('AntMaze1-v1')
test_env2 = gym.make(args.env_name)
test_env2.env_info = dict(goal_type='random')

test_env1 = record_video(test_env1,
                         'delta-s',
                         activate_per_episode=10,
                         name_prefix='farthest')
test_env2 = record_video(test_env2,
                         'delta-s',
                         activate_per_episode=10,
                         name_prefix='random')
test_env1, test_env2 = make_envs([test_env1, test_env2], seed=RANDOM_SEED)

agent = OfflineAgent(
    DeltaS(test_env1.observation_space['observation'].shape[0],
           test_env1.observation_space['desired_goal'].shape[0] if 'Fall' not in args.env_name else 3,
           test_env1.action_space.shape[0], 16), Preprocess())

agent.set_algm_reporter(get_reporter(agent.name))

offline_train_and_eval(agent,
                       dict(dataset=dataset), [test_env1, test_env2],
                       single_train_frames=50 * 256,
                       eval_per_train=2,
                       total_train_frames=999 * 3 * 100 * 256)

dataset.close()