# %%
import setup
from algorithm import DQNAlgorithm, Preprocess, DDQNAlgorithm
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

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# %%
env = gym.make("PongDeterministic-v4")
env.seed(RANDOM_SEED)
env.reset()
TOTAL_ACTIONS = env.action_space.n


# %%
TOTAL_ACTIONS


# %%
env = PreprocessObservation(env)
# env = GrayScaleObservation(env)
# env = ResizeObservation(env, 84)
env = FrameStack(env, num_stack=4)
env


# %%
DEFAULT_TRAINING_TIMES = 50_00_0000


# %%
TRAINING_TIMES = DEFAULT_TRAINING_TIMES

agent = Agent(env, DDQNAlgorithm(TOTAL_ACTIONS), Preprocess())
print(f"agent name: {agent.name}")
print(f"agent update_target: {agent.algm.update_target}")
training_rwds: List[int] = []

max_decry_times = 100_0000
with tqdm(total=DEFAULT_TRAINING_TIMES) as pbar:
    frames = 0
    while frames < TRAINING_TIMES:
        agent.reset()
        i = 0
        end = False
        while not end and frames < TRAINING_TIMES:

            (_, end) = agent.step()
            i += 1

        frames += i
        pbar.update(i)

        sigma = 1 - 0.95 / max_decry_times * \
            np.min([agent.algm.times, max_decry_times])

        training_rwds.append(np.sum([r for r in agent.episode_reward]))
        pbar.set_postfix(
            rwd=training_rwds[-1],
            sigma=sigma,
            memory_ratio=len(agent.algm.replay_memory) / 25_0000,
            loss=agent.algm.loss,
        )

        if frames >= 3_00_0000:
            print("reached 3_00_0000 frames, end!")
            break


# %%
np.save(f"./training_rwds_{agent.name}.arr", np.asarray(training_rwds))
torch.save(
    agent.algm.policy_network.state_dict(
    ), f"./policy_network_{agent.name}.params"
)
torch.save(
    agent.algm.target_network.state_dict(
    ), f"./target_network_{agent.name}.params"
)
# np.save("./training_loss.arr", np.asarray(agent.algm.loss))

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
    i = 1

    while not end and i < MAX_EPISODE_LENGTH:
        (o, end) = agent.step()
        i += 1
        env.render()
    rwds.append(np.sum([r for r in agent.episode_reward]))

np.save(f"./eval_rwds_{agent.name}.arr", np.asarray(rwds))
