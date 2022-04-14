import setup
from utils.common import ActionInfo, StepGeneric, Episode, TransitionGeneric
from torch import nn
import math
from collections import deque
import torch
from utils.preprocess import PreprocessInterface
from utils.algorithm import AlgorithmInterface
import plotly.graph_objects as go
from tqdm.autonotebook import tqdm
from torchvision import transforms as T
from utils.agent import Agent
from gym.spaces import Box
from typing import List, Tuple, Literal, Any, Optional, cast, Callable, Union, Iterable, Dict
import gym
import numpy.typing as npt
from utils.env import PreprocessObservation, FrameStack
import numpy as np

Observation = torch.Tensor
Action = int

State = torch.Tensor
Reward = int

Transition = TransitionGeneric[State, Action]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DQN(nn.Module):
    def __init__(self, n_actions: int):
        super(DQN, self).__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, (8, 8), 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: State) -> torch.Tensor:
        rlt = cast(torch.Tensor, self.net(x.to(DEVICE)))
        assert rlt.shape == (x.shape[0], self.n_actions)
        return rlt.cpu()


class DQNAlgorithm(AlgorithmInterface[State, Action]):
    def __init__(self, n_actions: int, gamma: float = 0.99):
        self.name = "dqn"
        self.n_actions = n_actions

        self.times = 0

        self.policy_network = DQN(n_actions).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=1e-4)

        self.target_network = DQN(n_actions).to(DEVICE)
        self.target_network.load_state_dict(self.policy_network.state_dict())

        for p in self.target_network.parameters():
            p.requires_grad = False

        self.target_network.eval()

        self.batch_size = 32

        self.update_target = 250

        self.replay_memory: deque[Transition] = deque(maxlen=math.ceil(25_0000))

        self.gamma = gamma
        self.loss_func = torch.nn.MSELoss()

        self.update_times = 4

        self.loss: float = -1.0

    def on_reset(self):
        pass
    

    def allowed_actions(self, _: State) -> List[Action]:
        return list(range(self.n_actions))

    def take_action(self, state: State) -> Action:
        rand = np.random.random()
        max_decry_times = 100_0000
        sigma = 1 - 0.95 / max_decry_times * np.min([self.times, max_decry_times])
        if rand < sigma:
            return np.random.choice(self.allowed_actions(state))

        else:
            act_vals: torch.Tensor = self.policy_network(
                self.resolve_lazy_frames(state)
            )
            maxi = torch.argmax(act_vals)
            return cast(int, maxi.item())

    def after_step(
        self,
        sar: Tuple[State, ActionInfo[Action], Reward],
        sa: Tuple[State, Optional[ActionInfo[Action]]],
    ):
        (s, a, r) = sar
        (sn, an) = sa
        self.replay_memory.append((s, a, r, sn, an))

        if self.times != 0 and self.times % (self.update_times) == 0:

            if len(self.replay_memory) >= 5 * self.batch_size:

                batch: List[Transition] = []
                for i in np.random.choice(len(self.replay_memory), self.batch_size):
                    batch.append(self.replay_memory[i])

                self.train(batch)

        if (
            self.times != 0
            and self.times % (self.update_target * self.update_times) == 0
        ):
            self.update_target_network()

        self.times += 1

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def resolve_lazy_frames(self, s: State) -> torch.Tensor:
        rlt = torch.cat([s[0], s[1], s[2], s[3]]).unsqueeze(0)
        return rlt

    def train(self, batch: List[Transition]):

        masks = torch.tensor(
            [0 if an is None else 1 for (_, _, _, _, an) in batch],
            dtype=torch.float,
        )

        target = torch.tensor(
            [r for (_, _, r, _, _) in batch], dtype=torch.float
        ) + masks * self.gamma * (
            torch.max(
                self.target_network(
                    torch.cat(
                        [self.resolve_lazy_frames(sn) for (_, _, _, sn, _) in batch]
                    )
                ).detach(),
                dim=1,
            )[0]
        )

        assert target.shape == (32,)
        s_curr = torch.cat([self.resolve_lazy_frames(s) for (s, _, _, _, _) in batch])
        assert s_curr.shape == (32, 4, 84, 84)

        x_vals = self.policy_network(s_curr)

        x = x_vals.gather(
            1, torch.tensor([a for (_, (a, _), _, _, _) in batch]).unsqueeze(1)
        ).squeeze(1)

        assert x.shape == (32,)

        loss = self.loss_func(x, target)
        self.loss = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_network.parameters():  # gradient clipping
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def on_termination(
        self, sar: Tuple[List[State], List[ActionInfo[Action]], List[Reward]]
    ):
        (s, a, r) = sar
        assert len(s) == len(a) + 1
        assert len(s) == len(r) + 1
        pass


class DDQNAlgorithm(DQNAlgorithm, AlgorithmInterface[State, Action]):
    def __init__(self, n_actions: int, gamma: float = 0.99):
        super().__init__(n_actions, gamma)
        self.update_target = 1250
        self.name = "ddqn"

    def train(self, batch: List[Transition]):

        s_next = torch.cat([self.resolve_lazy_frames(sn) for (_, _, _, sn, _) in batch])
        assert s_next.shape == (32, 4, 84, 84)

        q_next = self.target_network(s_next).detach()

        assert q_next.shape == (32, self.n_actions)

        masks = torch.tensor(
            [0 if an is None else 1 for (_, _, _, _, an) in batch],
            dtype=torch.float,
        )

        target = torch.tensor(
            [r for (_, _, r, _, _) in batch], dtype=torch.float
        ) + masks * self.gamma * q_next.gather(
            1, torch.argmax(self.policy_network(s_next), dim=1, keepdim=True)
        ).squeeze(
            1
        )

        assert target.shape == (32,)
        s_curr = torch.cat([self.resolve_lazy_frames(s) for (s, _, _, _, _) in batch])
        assert s_curr.shape == (32, 4, 84, 84)

        x_vals = self.policy_network(s_curr)

        x = x_vals.gather(
            1, torch.tensor([a for (_, (a, _), _, _, _) in batch]).unsqueeze(1)
        ).squeeze(1)

        assert x.shape == (32,)

        loss = self.loss_func(x, target)
        self.loss = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_network.parameters():  # gradient clipping
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


class Preprocess(PreprocessInterface[Observation, Action, State]):
    def __init__(self):
        pass

    def on_reset(self):
        pass

    def get_current_state(self, h: List[Observation]) -> State:
        assert len(h) > 0

        assert h[-1].shape == (4, 1, 84, 84)
        return h[-1]


class RandomAlgorithm(AlgorithmInterface[State, Action]):
    def __init__(self, n_actions: int):
        self.name = "random"
        self.n_actions = n_actions
        self.times = -1
        self.reset()

    def reset(self):
        self.last_action = None

    def allowed_actions(self, state: State) -> List[Action]:
        return list(range(self.n_actions))

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
        sar: Tuple[State, Action, Reward],
        sa: Tuple[State, Optional[Action]],
    ):
        pass
