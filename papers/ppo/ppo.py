# %%
from algorithm import PPO
import setup
from algorithm import RandomAlgorithm, Preprocess
import torch
import plotly.graph_objects as go
from tqdm.autonotebook import tqdm
from utils.agent import Agent
from typing import Dict, List, Any
import gym
from utils.env import PreprocessObservation, FrameStack, ToTensorEnv
from utils.env_sb3 import WarpFrame, MaxAndSkipEnv, NoopResetEnv, EpisodicLifeEnv
import numpy as np
from torch.utils.tensorboard import SummaryWriter


# %%
RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
# # torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# %%
env = gym.make("PongDeterministic-v4")
env.seed(RANDOM_SEED)
env.action_space.seed(RANDOM_SEED)
env.observation_space.seed(RANDOM_SEED)
env.reset()
TOTAL_ACTIONS = env.action_space.n
writer = SummaryWriter()


# %%
TOTAL_ACTIONS


# %%
# env = PreprocessObservation(env)
env = WarpFrame(env)
env = ToTensorEnv(env)
env = FrameStack(env, num_stack=4)
env


# %%
TRAINING_TIMES = 10_00_0000

j = 0


def report(info: Dict[Any, Any]):
    global j
    writer.add_scalar("target", info['target'], j)
    writer.add_scalar("policy_loss", info['policy_loss'], j)
    writer.add_scalar("entropy", info['entropy'], j)
    writer.add_scalar("value_loss", info['value_loss'], j)
    j += 1


agent = Agent(env, PPO(TOTAL_ACTIONS), Preprocess())
agent.set_algm_reporter(report)
# training_rwds: List[int] = []
print(f"agent name: {agent.name}")

with tqdm(total=TRAINING_TIMES) as pbar:
    frames = 0
    epi = 0
    while frames < TRAINING_TIMES:
        agent.reset()
        i = 0
        end = False
        while not end and frames < TRAINING_TIMES:

            (_, end) = agent.step()
            i += 1
            # agent.render('human')
        epi += 1

        frames += i
        pbar.update(i)

        rwd = np.sum([r for r in agent.episode_reward])
        writer.add_scalar("reward", rwd, epi)

        if frames >= 3_00_0000:
            print("reached 3_00_0000 frames, end!")
            break

torch.save(agent.algm.network.state_dict(), f"./{agent.name}_network.params")


# %%
EVALUATION_TIMES = 30
MAX_EPISODE_LENGTH = 18_000
rwds: List[int] = []
agent.toggleEval(True)

# eval_env = gym.make("PongDeterministic-v4")
# eval_env.seed(RANDOM_SEED)
# eval_env = WarpFrame(eval_env)
# eval_env = ToTensorEnv(eval_env)
# eval_env = FrameStack(eval_env, num_stack=4)
# eval_env.reset()

# agent.env = eval_env

for _ in tqdm(range(EVALUATION_TIMES)):
    agent.reset()

    end = False
    i = 0

    while not end and i < MAX_EPISODE_LENGTH:
        (o, end) = agent.step()
        i += 1
        agent.render('human')
    rwds.append(np.sum([r for r in agent.episode_reward]))

np.save(f"./eval_rwds_{agent.name}.arr", np.asarray(rwds))

# %%
