import setup
from papers.hiro.hiro import Hiro, Preprocess
from utils.agent import Agent
import gym
from setup import RANDOM_SEED
from utils.reporter import get_reporter
from utils.env import train_and_eval, make_train_and_eval_env
from utils.env_sb3 import RecordVideo, RescaleAction
from envs.create_maze_env import create_maze_env
from envs import EnvWithGoal

# %%
eval_env = train_env = EnvWithGoal(create_maze_env('AntMaze'), 'AntMaze')
# eval_env = RecordVideo(eval_env,
#                        'vlog/lesson',
#                        episode_trigger=lambda episode_id: episode_id % 5 == 0,
#                        name_prefix='lesson')
train_env, eval_env = make_train_and_eval_env((train_env, eval_env), [],
                                              RANDOM_SEED)

# %%

agent = Agent(
    train_env,
    Hiro(31, 2,
         train_env.action_space.shape[0]), Preprocess())

agent.set_algm_reporter(get_reporter(agent.name))

train_and_eval(agent, eval_env)
