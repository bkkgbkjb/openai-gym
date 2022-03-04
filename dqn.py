# %%
import numpy as np
import numpy.typing as npt
import gym

from typing import List, Tuple, Literal, Any, Optional, cast, Callable
import plotly.graph_objects as go
from utils.agent import Agent
from tqdm.autonotebook import tqdm
from utils.algorithm import AlgorithmInterface
from utils.preprocess import PreprocessInterface
import torch
from collections import deque
from torchvision import transforms
import math
from torch import nn
import sys
from copy import deepcopy
from utils.common import Step, Episode, TransitionGeneric


# %%
RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# %%
env = gym.make("Pong-v0")
env.seed(RANDOM_SEED)
env.reset()
env._max_episode_steps = 1_0000
TOTAL_ACTIONS = env.action_space.n


# %%
TOTAL_ACTIONS

# %%
# shape is (210, 160, 3)
Observation = npt.NDArray[np.uint8]
Action = int

# shape is (4, 210, 160, 3)
State = torch.Tensor
Reward = int

Transition = TransitionGeneric[State, Action]

# %%


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, (8, 8), 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.Linear(512, TOTAL_ACTIONS),
        ).to(device)

    def forward(self, x: State) -> torch.Tensor:
        rlt = cast(torch.Tensor, self.net(x.to(device)))
        assert rlt.shape == (x.shape[0], TOTAL_ACTIONS)
        return rlt.cpu()


# %%
class RandomAlgorithm(AlgorithmInterface[State, Action]):
    def __init__(self):

        self.times = 1
        self.last_action = None

    def reset(self):
        pass

    def allowed_actions(self, state: State) -> List[Action]:
        return list(range(TOTAL_ACTIONS))

    def take_action(self, state: State) -> Action:
        self.times += 1

        if self.times % 10 == 0:
            act = np.random.choice(self.allowed_actions(state))
            self.last_action = act
            return act

        if self.last_action is not None:
            return self.last_action

        act = np.random.choice(self.allowed_actions(state))
        self.last_action = act
        return act

    def after_step(
        self,
        sa: Tuple[State, Action],
        episode: Episode[State, Action],
    ):
        pass

    def on_termination(self, episode: Episode[State, Action]):
        pass

# %%


class NNAlgorithm(AlgorithmInterface[State, Action]):
    def __init__(self, nn: DQN, training_times: int = 50_00_0000, gamma: float = 0.99):
        self.network = nn
        self.optimizer = torch.optim.RMSprop(
            self.network.parameters(), 1e-3, 0.95, 0.01
        )

        self.shrink = min(training_times / 50_00_0000, 1)
        if self.shrink != 1:
            print(f"training on shrinked mode: {self.shrink}")

        self.target_network = DQN()
        self.target_network.load_state_dict(self.network.state_dict())

        self.times: int = 1
        self.batch_size = 32

        self.update_freq: int = 5
        self.update_target = 100

        self.memory_replay: deque[Transition] = deque(
            maxlen=math.ceil(100_0000 / 5 * self.shrink)
        )
        self.gamma = gamma
        self.loss_func = torch.nn.MSELoss()

    def reset(self):
        pass

    def allowed_actions(self, _: State) -> List[Action]:
        return list(range(TOTAL_ACTIONS))

    def take_action(self, state: State) -> Action:
        rand = np.random.random()
        max_decry_times = 100_0000 * self.shrink
        sigma = 1 - 0.9 / max_decry_times * \
            np.min([self.times, max_decry_times])
        if rand < sigma:
            return np.random.choice(self.allowed_actions(state))
        else:
            act_vals: torch.Tensor = self.network(state)
            maxi = torch.argmax(act_vals)
            return cast(Action, maxi)

    def after_step(
        self,
        sa: Tuple[State, Optional[Action]],
        episode: Episode[State, Action],
    ):
        (s, a, r) = episode[-1]
        (sn, an) = sa
        self.memory_replay.append((s, cast(Action, a), cast(float, r), sn, an))

        if self.times % self.update_freq == 0 and len(self.memory_replay) >= 48:

            batch: List[Transition] = []
            for i in np.random.randint(0, len(self.memory_replay), 32):
                batch.append(self.memory_replay[i])

            self.train(batch)

        if self.times % (self.update_target * self.update_freq) == 0:
            self.update_target_network()

        self.times += 1

    def on_termination(self, episode: Episode[State, Action]):
        pass

    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())
        # pass

    def clip_reward(self, r: float) -> float:
        if r > 0:
            return 1.0
        elif r < 0:
            return -1.0
        else:
            return 0

    def train(self, batch: List[Transition]):
        # target = torch.tensor(
        #     [
        #         self.clip_reward(r)
        #         if an is None
        #         else self.clip_reward(r)
        #         + self.gamma * torch.max(self.target_network(sn))
        #         for (_, _, r, sn, an) in batch
        #     ]
        # )

        masks = torch.tensor(
            [0 if an is None else 1 for (_, _, _, _, an) in batch],
            dtype=torch.float,
        )

        target = torch.tensor(
            [self.clip_reward(r) for (_, _, r, _, _) in batch], dtype=torch.float
        ) + torch.inner(
            masks,
            self.gamma
            * torch.max(
                self.network(torch.cat([sn for (_, _, _, sn, _) in batch])), dim=1
            )[0],
        )

        assert target.shape == (32,)
        x_vals = self.network(torch.cat([s for (s, _, _, _, _) in batch]))
        x = x_vals[range(x_vals.shape[0]), [a for (_, a, _, _, _) in batch]]

        assert x.shape == (32,)

        loss = self.loss_func(x, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Preprocess(PreprocessInterface[Observation, Action, State]):
    def __init__(self):
        self.trfm: Callable[[Observation], State] = transforms.Compose(
            [transforms.ToTensor(), transforms.Grayscale(),
             transforms.Resize((84, 84))]
        )
        self.reset()
        # self.history: Episode[State, Action] = []

    def reset(self):
        self.history: Episode[State, Action] = []

    def get_current_state(self, h: Episode[Observation, Action]) -> State:
        assert len(h) > 0

        last_4_arr = self.stack_4(h, -1)

        rlt = torch.stack([self.trfm(i)
                          for i in last_4_arr]).squeeze(1).unsqueeze(0)
        assert rlt.shape == (1, 4, 84, 84)
        return rlt

    def stack_4(
        self, h: Episode[Observation, Action], idx: int
    ) -> npt.NDArray[np.uint8]:

        assert idx < 0
        last_4_index = [-12 + idx, -8 + idx, -4 + idx, idx]

        last_4: List[Observation] = []
        for idx in last_4_index:
            if -idx <= len(h):
                last_4.append(np.asarray((h[idx][0])))

        last_4_arr = np.asarray(last_4)
        while last_4_arr.shape[0] < 4:
            last_4_arr = np.insert(last_4_arr, 0, last_4[0], axis=0)

        assert last_4_arr.shape == (4, 210, 160, 3)

        return last_4_arr

    def transform_history(
        self, h: Episode[Observation, Action]
    ) -> Episode[State, Action]:
        delta = len(h) - len(self.history)
        assert delta == 1

        (_, a, r) = h[-1]
        last_4_arr = self.stack_4(h, -1)
        s = torch.stack([self.trfm(i)
                        for i in last_4_arr]).squeeze(1).unsqueeze(0)
        assert s.shape == (1, 4, 84, 84)
        self.history.append((s, a, r))

        return self.history


# %%
TRAINING_TIMES = 50_00_0000
# TRAINING_TIMES = 2_0000

agent = Agent(env, NNAlgorithm(DQN(), TRAINING_TIMES), Preprocess())
training_rwds: List[float] = []

with tqdm(total=TRAINING_TIMES) as pbar:
    pbar.update(1)
    frames = 1
    while frames < TRAINING_TIMES:
        agent.reset(["preprocess"])

        end = False

        while not end and frames < TRAINING_TIMES:
            (o, end, episode) = agent.step()

            frames += 1
            pbar.update(1)

        training_rwds.append(
            np.sum([r if r is not None else 0 for (_, _, r) in agent.episode])
        )


# %%
np.save("./training.arr", np.asarray(training_rwds))


# %%
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=[i + 1 for i in range(len(training_rwds))],
               y=[r for r in training_rwds])
)
# fig.update_yaxes(type="log")
# fig.update_layout(yaxis_type="log")
fig.show()


# %%
EVALUATION_TIMES = 30
MAX_EPISODE_LENGTH = 18_000
rwds: List[int] = []
agent.toggleEval(False)

for _ in tqdm(range(EVALUATION_TIMES)):
    agent.reset(['preprocess'])

    end = False
    i = 1

    while not end and i < MAX_EPISODE_LENGTH:
        (o, end, episode) = agent.step()
        i += 1
        # env.render()
        # if end:
        #     rwds.append(np.sum([r if r is not None else 0 for (_,
        #                                                        _, r) in cast(Episode, episode)]))
    rwds.append(
        np.sum([r if r is not None else 0 for (_, _, r) in agent.episode]))


# %%
np.save("./eval.arr", np.asarray(rwds))

# %%
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=[i + 1 for i in range(len(rwds))],
               y=[r for r in rwds])
)
# fig.update_yaxes(type="log")
# fig.update_layout(yaxis_type="log")
fig.show()
