import setup
import gym
import mujoco_maze
from utils.agent import Agent
from utils.env import glance
from papers.sac.algorithm import NewSAC, Preprocess
from setup import RANDOM_SEED
from utils.reporter import get_reporter
from utils.env import train_and_eval, make_train_and_eval_env
from utils.env_sb3 import RecordVideo

train_env = gym.make('Ant-v3')
eval_env = gym.make('Ant-v3')
eval_env = RecordVideo(eval_env,
                       'vlog/sac-on-ant',
                       episode_trigger=lambda episode_id: episode_id % 5 == 0,
                       name_prefix='sac-on-ant')

train_env, eval_env = make_train_and_eval_env((train_env, eval_env), [],
                                              RANDOM_SEED)

# %%

agent = Agent(
    train_env,
    NewSAC(train_env.observation_space.shape[0],
           train_env.action_space.shape[0], 1), Preprocess())

agent.set_algm_reporter(get_reporter(agent.name))

train_and_eval(agent, eval_env, total_train_frames=int(3e6))
