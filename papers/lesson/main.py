import setup
from lesson import LESSON, Preprocess, MAX_TIMESTEPS
from utils.agent import Agent
import gym
from setup import RANDOM_SEED
from utils.reporter import get_reporter
from utils.env import train_and_eval, make_train_and_eval_env
from goal_env.mujoco import *
from utils.env_sb3 import RecordVideo, RescaleAction

# %%
train_env = gym.make("AntMaze1-v1")
train_env = RecordVideo(
    train_env,
    'vlog/lesson',
    episode_trigger= lambda episode_id: episode_id % 25 ==0,
    name_prefix='lesson-train'
)
eval_env = gym.make("AntMaze1Test-v1")
eval_env = RecordVideo(
    eval_env,
    "vlog/lesson",
    episode_trigger=lambda episode_id: episode_id % 5 == 0,
    name_prefix="lesson-eval",
)
train_env, eval_env = make_train_and_eval_env((train_env, eval_env), [], RANDOM_SEED)

# %%

agent = Agent(
    LESSON(
        train_env.observation_space.shape[0],
        train_env.goal_space.shape[0],
        train_env.action_space.shape[0],
    ),
    Preprocess(),
)

agent.set_algm_reporter(get_reporter(agent.name))

train_and_eval(agent, train_env, eval_env, total_train_frames=int(1e7), eval_per_train=5)
