import setup
from utils.common import ActionInfo, Step, Episode, TransitionGeneric
from torch import nn
import math
from collections import deque
import torch
from utils.preprocess import PreprocessInterface
from utils.algorithm import AlgorithmInterface
import plotly.graph_objects as go
from torch.distributions import Categorical
from tqdm.autonotebook import tqdm

from torchvision import transforms as T
from utils.agent import Agent
from gym.spaces import Box
from typing import List, Tuple, Literal, Any, Optional, cast, Callable, Union, Iterable
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


class Actor(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()

        self.n_actions = n_actions

        self.network = nn.Sequential(
            nn.Conv2d(4, 32, (8, 8), 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), 2),
            nn.ReLU(),
            nn.Conv2d(64, 32, (3, 3), 1),
            nn.Flatten(),
            nn.Linear(7 * 7 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
            nn.Softmax(dim=0),
        )

    def forward(self, s: State) -> torch.Tensor:
        rlt = cast(torch.Tensor, self.network(s.to(DEVICE)))
        assert rlt.shape == (s.size(0), self.n_actions)
        return rlt


class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(4, 32, (8, 8), 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), 2),
            nn.ReLU(),
            nn.Conv2d(64, 32, (3, 3), 1),
            nn.Flatten(),
            nn.Linear(7 * 7 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, s: State) -> torch.Tensor:
        rlt = cast(torch.Tensor, self.network(s.to(DEVICE)))
        assert rlt.shape == (s.size(0), 1)
        return rlt


class PPO(AlgorithmInterface[State, Action]):
    def __init__(self, n_actions: int, gamma: float = 0.99):
        self.frame_skip = 0
        self.name = "ppo"
        self.n_actions = n_actions
        self.actor = Actor(n_actions).to(DEVICE)
        self.critic = Critic().to(DEVICE)
        self.times = -1

        self.gamma = gamma
        self.update_freq = 250

    def allowed_actions(self, state: State) -> List[Action]:
        return list(range(self.n_actions))

    def resolve_lazy_frames(self, s: State) -> torch.Tensor:
        rlt = torch.cat([s[0], s[1], s[2], s[3]]).unsqueeze(0)
        return rlt

    def take_action(self, state: State) -> ActionInfo[Action]:
        with torch.no_grad():

            act_probs = self.actor(self.resolve_lazy_frames(state))
            dist = Categorical(act_probs)
            act = dist.sample()

            return (cast(int, act.item()), {"log_prob": dist.log_prob(act)})

    def after_step(
        self,
        sar: Tuple[State, ActionInfo[Action], Reward],
        sa: Tuple[State, Optional[ActionInfo[Action]]],
    ):
        self.times += 1
        (s, a, r) = sar
        (sn, an) = sa

        self.memory.append((s, a, r, sn, an))

        if self.times != 0 and self.times % self.update_freq == 0:
            self.train()

        pass

    def train(self):
        advs: List[float] = []

        T = len(self.memory)
        (_, _, _, st, at) = self.memory[-1]
        assert at is not None

        for t, (s, _, _, _, _) in enumerate(self.memory):
            _adv = -self.critic(s)

            for j in range(t, T):
                (_, _, r, _, _) = self.memory[j]
                _adv += self.gamma ** (j - t) * r

            _adv += self.gamma ** (T - t) * self.critic(st)

            advs.append(_adv)

        assert len(advs) == len(self.memory)

        ratios = [
            math.exp(Categorical(self.actor(s)).log_prob(a) - info["log_prob"])
            for (s, (a, info), _, _, _) in self.memory
        ]
        assert len(ratios) == len(self.memory)

        # (_, _, _, sl, (al, infol)) = self.memory[-1]
        # ratios.append(
        #     math.exp(Categorical(self.actor(sl)).log_prob(al) - infol["log_prob"])
        # )

    def on_termination(
        self, sar: Tuple[List[State], List[ActionInfo[Action]], List[Reward]]
    ):
        (s, a, r) = sar
        assert len(s) == len(a) + 1
        assert len(s) == len(r) + 1
        pass

    def reset(self):
        self.memory: List[Transition] = []


class RandomAlgorithm(AlgorithmInterface[State, Action]):
    def __init__(self, n_actions: int):
        self.name = "random"
        self.frame_skip = 0
        self.n_actions = n_actions
        self.times = -1
        self.action_keeps = 10
        self.reset()

    def reset(self):
        self.last_action = None

    def allowed_actions(self, state: State) -> List[Action]:
        return list(range(self.n_actions))

    def take_action(self, state: State) -> Action:
        self.times += 1

        if self.times % self.action_keeps == 0:
            act = np.random.choice(self.allowed_actions(state))
            self.last_action = act
            return act

        assert self.last_action is not None
        return self.last_action

    def after_step(
        self,
        sar: Tuple[State, Action, Reward],
        sa: Tuple[State, Optional[Action]],
    ):
        pass

    def on_termination(self, sar: Tuple[List[State], List[Action], List[Reward]]):
        (s, a, r) = sar
        assert len(s) == len(a) + 1
        assert len(s) == len(r) + 1
        pass


class Preprocess(PreprocessInterface[Observation, Action, State]):
    def __init__(self):
        self.reset()

    def reset(self):
        pass

    def get_current_state(self, h: List[Observation]) -> State:
        assert len(h) > 0

        assert h[-1].shape == (4, 1, 84, 84)
        return h[-1]
