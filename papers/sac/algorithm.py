from cupshelpers import Device
from cv2 import log
import setup
from utils.common import (
    ActionInfo,
    StepGeneric,
    Episode,
    TransitionGeneric,
    NotNoneStepGeneric,
)
from torch import nn
import math
from collections import deque
import torch
from utils.preprocess import PreprocessInterface
from utils.algorithm import AlgorithmInterface
import plotly.graph_objects as go
from torch.distributions import Categorical, Normal
from tqdm.autonotebook import tqdm

from torchvision import transforms as T
from utils.agent import Agent
from gym.spaces import Box
from typing import List, Tuple, Literal, Any, Optional, cast, Callable, Union, Iterable, Dict
import gym
import numpy.typing as npt
from utils.env import PreprocessObservation, FrameStack, resolve_lazy_frames
import numpy as np

Observation = torch.Tensor
Action = torch.Tensor

State = Observation
Reward = int

Transition = TransitionGeneric[State, Action]
Step = StepGeneric[State, ActionInfo[Action]]
NotNoneStep = NotNoneStepGeneric[State, ActionInfo[Action]]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class VFunction(nn.Module):
    def __init__(self, n_state: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(n_state, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1))
        ).to(DEVICE)

    def forward(self, s: State) -> torch.Tensor:
        return self.net(s.to(DEVICE))


class QFunction(nn.Module):
    def __init__(self, n_state: int, n_action: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(n_state + n_action, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1))
        ).to(DEVICE)

    def forward(self, s: State, a: Action) -> torch.Tensor:
        assert s.size(1) == 17
        assert a.size(1) == 6
        return self.net(torch.cat([s.to(DEVICE), a.to(DEVICE)], 1))


class PaiFunction(nn.Module):
    def __init__(self, n_state: int, n_action):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(n_state, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
        ).to(DEVICE)

        self.mean = layer_init(nn.Linear(256, n_action)).to(DEVICE)
        self.std = layer_init(nn.Linear(256, n_action)).to(DEVICE)

    def forward(self, s: State) -> Tuple[torch.Tensor, torch.Tensor]:
        assert s.size(1) == 17
        x = self.net(s.to(DEVICE))
        mean = self.mean(x)
        log_std = self.std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)

        assert mean.shape == (s.size(0), 6)
        assert log_std.shape == (s.size(0), 6)

        return mean, log_std

    def sample(self, s: State):
        mean, log_std = self.forward(s)
        std = log_std.exp()
        normal = Normal(mean, std)
        raw_act = normal.rsample()
        raw_log_prob = normal.log_prob(raw_act)

        assert raw_act.shape == (s.size(0), 6)
        assert raw_log_prob == (s.size(0), 1)

        act = torch.tanh(raw_act)

        mod_log_prob = (1 - act.pow(2) + 1e-6).log().sum(1, keepdim=True)
        assert mod_log_prob == (s.size(0), 1)
        log_prob = raw_log_prob - mod_log_prob

        mean = torch.tanh(mean)
        return act, log_prob, mean


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SAC(AlgorithmInterface[State, Action]):
    def __init__(self, n_state: int, n_actions: int):
        self.name = 'sac'
        self.n_actions = n_actions
        self.n_state = n_state

        self.online_v = VFunction(self.n_state)
        self.old_v = VFunction(self.n_state)
        self.old_v.load_state_dict(self.online_v.state_dict())

        self.policy = PaiFunction(self.n_state, self.n_actions)

        self.q1 = QFunction(self.n_state, self.n_actions)
        self.q2 = QFunction(self.n_state, self.n_actions)

        self.gamma = 0.99

        self.reset()

    def reset(self):
        self.times = 0
        self.replay_memory: deque[Transition] = deque(maxlen=100_0000)

    def on_agent_reset(self):
        pass

    def take_action(self, state: State) -> Action:
        return torch.as_tensor([0.0] * 6)

    def on_episode_termination(self, sar: Tuple[List[State], List[ActionInfo[Action]], List[Reward]]):
        pass

    def after_step(self, sar: Tuple[State, ActionInfo[Action], Reward], sa: Tuple[State, Optional[ActionInfo[Action]]]):
        (s, a, r) = sar
        (sn, an) = sa
        self.replay_memory.append((s, a, r, sn, an))
        self.times += 1


class Preprocess(PreprocessInterface[Observation, Action, State]):
    def __init__(self):
        pass

    def on_agent_reset(self):
        pass

    def get_current_state(self, h: List[Observation]) -> State:
        assert len(h) > 0

        # assert h[-1].shape == (4, 1, 84, 84)
        return h[-1]
