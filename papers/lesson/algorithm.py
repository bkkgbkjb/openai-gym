import setup
from utils.transition import (
    Transition,
    TransitionTuple,
    resolve_transitions,
)
from torch import nn
from collections import deque
import torch
from utils.preprocess import PreprocessI
from utils.algorithm import Algorithm
from torch.distributions import Categorical, Normal
from typing import Union
from utils.nets import NeuralNetworks, layer_init

from typing import List, Tuple, Any, Optional, Callable, Dict
import numpy as np

from utils.replay_buffer import ReplayBuffer

Observation = torch.Tensor
Action = torch.Tensor

State = Observation
Goal = torch.Tensor
Reward = int

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Preprocess(PreprocessI[Observation, State]):

    def __init__(self):
        pass

    def on_agent_reset(self):
        pass

    def get_current_state(self, h: List[Observation]) -> State:
        assert len(h) > 0

        # assert h[-1].shape == (4, 1, 84, 84)
        return torch.from_numpy(h[-1]).type(torch.float32).to(DEVICE)


class LowCritic(NeuralNetworks):

    def __init__(self, state_dim: int, goal_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim + goal_dim + action_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1)),
        ).to(DEVICE)

        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim

    def forward(self, s: State, g: Goal, a: Action) -> torch.Tensor:
        assert s.size(1) == self.state_dim
        assert a.size(1) == self.action_dim
        assert g.size(1) == self.goal_dim
        return self.net(
            torch.cat([s.to(DEVICE), g.to(DEVICE),
                       a.to(DEVICE)], 1))


class LowActor(NeuralNetworks):

    def __init__(
        self,
        state_dim: int,
        goal_dim,
        action_dim: int,
    ):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim + goal_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, action_dim)),
            nn.Tanh(),
        ).to(DEVICE)

        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.action_scale = action_scale

    def forward(self, s: State, g: torch.Tensor) -> torch.Tensor:
        assert s.size(1) == self.state_dim
        assert g.size(1) == self.goal_dim
        x = torch.cat([s, g], dim=1)
        return self.net(x)


class Representation(NeuralNetworks):

    def __init__(self, state_dim: int, goal_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim

        self.net = nn.Sequential(nn.Linear(self.state_dim, 100), nn.ReLU(),
                                 nn.Linear(100, 100), nn.ReLU(),
                                 nn.Linear(100, self.goal_dim))

    def forward(self, s: State):
        return self.net(s)


class LESSON(Algorithm):

    def __init__(self, state_dim: int, goal_dim: int, action_dim: int):
        self.name = "lesson"
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.goal_dim = goal_dim

        self.gamma = 0.99

        self.tau = 5e-3

        self.start_traininig_size = int(1e4)
        self.mini_batch_size = 256

        self.low_actor = LowActor(self.state_dim, self.goal_dim,
                                  self.action_dim)
        self.low_actor_target = self.low_actor.clone().no_grad()

        self.low_critic = LowCritic(self.state_dim, self.goal_dim,
                                    self.action_dim)
        self.low_critic_target = self.low_critic.clone().no_grad()

        self.low_critic_loss = nn.MSELoss()

        self.low_actor_optim = torch.optim.Adam(self.low_actor.parameters(),
                                                lr=2e-4)
        self.low_critic_optim = torch.optim.Adam(self.low_critic.parameters(),
                                                 lr=2e-4,
                                                 weight_decay=1e-5)

        self.reset()

    def reset(self):
        self.times = 0
        self.replay_memory = ReplayBuffer((self.state_dim, ),
                                          (self.action_dim, ))

    @torch.no_grad()
    def take_action(self, state: State, goal: Goal) -> Action:
        action = self.low_actor(state, goal)
        return action.detach().cpu().squeeze(0).numpy()

    def after_step(self, transition: TransitionTuple[State]):
        self.replay_memory.append(Transition(transition))

        if self.replay_memory.len >= self.start_traininig_size:
            self.train()

        self.times += 1

    def train(self):
        pass