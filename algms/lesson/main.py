import setup
from lesson import LESSON, Preprocess, MAX_TIMESTEPS
from utils.agent import Agent
import gym
from setup import RANDOM_SEED
from utils.reporter import get_reporter
from utils.env import make_envs, record_video, train_and_eval, make_train_and_eval_env
from goal_env.mujoco import *
from utils.env_sb3 import RecordVideo, RescaleAction
from datetime import datetime

# %%
train_env = gym.make("AntMaze1-v1")
train_env._env_id = 'train'

agent = Agent(
    LESSON(
        train_env.observation_space.shape[0],
        train_env.goal_space.shape[0],
        train_env.action_space.shape[0],
    ),
    Preprocess(),
)

eval_env1 = gym.make("AntMaze1Test-v1")
eval_env2 = gym.make("AntMaze1-v1")

eval_env1 = record_video(eval_env1, agent.name, 5, name_prefix='eval')
eval_env1._env_id = 'eval_farthest'
eval_env2 = record_video(eval_env2, agent.name, 5, name_prefix='train')
eval_env2._env_id = 'eval_random'

train_env, eval_env1, eval_env2 = make_envs([train_env, eval_env1, eval_env2])

agent.set_algm_reporter(get_reporter(agent.name))

train_and_eval(agent,
               train_env, [eval_env1, eval_env2],
               total_train_frames=int(1e7),
               eval_per_train=5)
