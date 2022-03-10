# %%
import setup
from algorithm import RandomAlgorithm, Preprocess
import torch
import plotly.graph_objects as go
from tqdm.autonotebook import tqdm
from utils.agent import Agent
from typing import List
import gym
from utils.env import PreprocessObservation, FrameStack
import numpy as np


# %%
RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# %%
env = gym.make("PongDeterministic-v4")
env.seed()
env.reset()
TOTAL_ACTIONS = env.action_space.n


# %%
TOTAL_ACTIONS


# %%
env = PreprocessObservation(env)
env = FrameStack(env, num_stack=4)
env


# %%
TRAINING_TIMES = 1000

agent = Agent(env, RandomAlgorithm(TOTAL_ACTIONS), Preprocess())
training_rwds: List[int] = []
print(f"agent name: {agent.name}")

with tqdm(total=TRAINING_TIMES) as pbar:
    frames = 0
    while frames < TRAINING_TIMES:
        agent.reset()
        i = 0
        end = False
        while not end and frames < TRAINING_TIMES:

            (_, end) = agent.step()
            i += 1
            agent.render('human')


        frames += i
        pbar.update(i)

        if frames >= 3_00_0000:
            print("reached 3_00_0000 frames, end!")
            break


# %%
np.save(f"./training_rwds_{agent.name}.arr", np.asarray(training_rwds))

# %%
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=[i + 1 for i in range(len(training_rwds))], y=[r for r in training_rwds]
    )
)
fig.write_image(f"./rwd_img_{agent.name}.png")
fig.show()


# %%
EVALUATION_TIMES = 30
MAX_EPISODE_LENGTH = 18_000
rwds: List[int] = []
agent.toggleEval(True)

for _ in tqdm(range(EVALUATION_TIMES)):
    agent.reset()

    end = False
    i = 0

    while not end and i < MAX_EPISODE_LENGTH:
        (o, end) = agent.step()
        i += 1
        env.render()
    rwds.append(np.sum([r for r in agent.episode_reward]))

np.save(f"./eval_rwds_{agent.name}.arr", np.asarray(rwds))
