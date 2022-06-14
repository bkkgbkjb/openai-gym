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

j = 0


def every_five_episode(episode_id: int) -> bool:
    global j
    record = j % 5 == 0
    j += 1
    return record


train_env, eval_env = make_train_and_eval_env("AntUMaze-v0", [
    lambda env, kind: RecordVideo(
        env, 'vlog', episode_trigger=lambda episode_id: episode_id % 5 == 0, name_prefix='sac-not-work-on-ant-maze')
    if kind == 'eval' else env
], RANDOM_SEED)

# %%

agent = Agent(
    train_env,
    NewSAC(train_env.observation_space.shape[0],
           train_env.action_space.shape[0]), Preprocess())

agent.set_algm_reporter(get_reporter(agent.name))

train_and_eval(agent, eval_env)
