import setup
import torch
import numpy as np
from utils.agent import Agent, OfflineAgent
from algorithm import H, Preprocess
import gym
import d4rl
from setup import RANDOM_SEED
from utils.reporter import get_reporter
from args import args, exp_name
import h5py
from utils.env_sb3 import flat_to_episode
from utils.env import glance, make_envs, offline_train_and_eval, record_video
from teleport import Teleport

# %%
# env = gym.make("antmaze-umaze-diverse-v2")


[env] = make_envs(args.env)

# glance(env)

ds = env.get_dataset()

end_index = (np.argwhere(ds["timeouts"] == True))[-1].item() + 1

init = (0.0, 0.0)
# end = (15.0, 20.0)

states = ds["observations"][:end_index]
actions = ds["actions"][:end_index]
pos = ds["observations"][:end_index, :2]
goals = ds["infos/goal"][:end_index]
# rewards = [
#     1.0 if (np.linalg.norm(p - goals[i]) <= 0.5) else 0.0 for i, p in enumerate(pos)
# ]
rewards = ds["rewards"][:end_index]

dones = ds["timeouts"][:end_index]
infos = [dict(pos=pos[i], goal=goals[i]) for i in range(len(pos))]

episodes = flat_to_episode(states, actions, rewards, dones, infos)

algm = H(
    env.observation_space.shape[0],
    len(env.target_goal),
    env.action_space.shape[0],
    1,
)
agent = OfflineAgent(algm, Preprocess())

exp_name = exp_name or agent.name

agent.set_algm_reporter(get_reporter(exp_name))

if args.log_video:
    env = record_video(env, exp_name, activate_per_episode=5, name_prefix=exp_name)

env = Teleport(env, np.array([-999] * 6))

offline_train_and_eval(
    agent,
    dict(episodes=episodes),
    env,
    single_train_frames=100 * 1024,
    eval_per_train=3,
    eval_repeats=10,
    total_train_frames=999 * 6 * 100 * 1024,
)
