import setup
from hiro import Hiro, Preprocess
from utils.agent import Agent
import gym
from setup import RANDOM_SEED
from utils.reporter import get_reporter
from utils.env import glance, train_and_eval, make_train_and_eval_env
from utils.env_sb3 import RecordVideo, RescaleAction
from envs.create_maze_env import create_maze_env
from envs import EnvWithGoal
# from goal_env.mujoco import *

# %%
# train_env = gym.make('AntMaze1-v1')
# eval_env = gym.make('AntMaze1Test-v1')
eval_env = train_env = EnvWithGoal(create_maze_env('AntMaze'), 'AntMaze')
train_env = RecordVideo(
    train_env,
    'vlog/hiro',
    episode_trigger=lambda episode_id: episode_id % 25 == 0,
    name_prefix='hiro-train')
eval_env = RecordVideo(eval_env,
                       'vlog/hiro',
                       episode_trigger=lambda episode_id: episode_id % 5 == 0,
                       name_prefix='hiro-eval')
eval_env.evaluate = True

train_env, eval_env = make_train_and_eval_env((train_env, eval_env), [],
                                              RANDOM_SEED)
train_env.seed(0)
train_env.base_env.seed(0)
train_env.base_env.wrapped_env.seed(0)
eval_env.seed(5)
eval_env.base_env.seed(5)
eval_env.base_env.wrapped_env.seed(5)

# %%

agent = Agent(Hiro(31, 2, train_env.action_space.shape[0]), Preprocess())

agent.set_algm_reporter(get_reporter(agent.name))

train_and_eval(agent, train_env, eval_env, total_train_frames=int(1e7), eval_per_train=5)
