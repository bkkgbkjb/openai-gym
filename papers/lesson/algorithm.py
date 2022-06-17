import setup
from utils.common import Info, ActionInfo
from utils.step import NotNoneStep, Step
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
from papers.sac import NewSAC
import numpy as np

from utils.replay_buffer import ReplayBuffer

Observation = torch.Tensor
Action = np.ndarray

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

    def forward(self, s: State, g: Goal, a: torch.Tensor) -> torch.Tensor:
        assert s.size(1) == self.state_dim
        assert a.shape[0] == self.action_dim
        assert g.size(1) == self.goal_dim
        return self.net(
            torch.cat([s.to(DEVICE), g.to(DEVICE),
                       a.to(DEVICE)], 1))


class LowActor(NeuralNetworks):

    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
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

    def forward(self, s: State, g: torch.Tensor) -> torch.Tensor:
        assert s.size(1) == self.state_dim
        assert g.size(1) == self.goal_dim
        x = torch.cat([s, g], dim=1)
        return self.net(x)


class RepresentationNetwork(NeuralNetworks):

    def __init__(self, state_dim: int, goal_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim

        self.net = nn.Sequential(nn.Linear(self.state_dim, 100), nn.ReLU(),
                                 nn.Linear(100, 100), nn.ReLU(),
                                 nn.Linear(100, self.goal_dim))

    def forward(self, s: State):
        return self.net(s)


class LowNetwork(Algorithm):

    def __init__(self, state_dim: int, goal_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim

        self.actor = LowActor(self.state_dim, self.goal_dim, self.action_dim)
        self.actor_target = self.actor.clone().no_grad()

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=2e-4)

        self.critic = LowCritic(self.state_dim, self.goal_dim, self.action_dim)
        self.critic_target = self.critic.clone().no_grad()

        self.critic_loss = nn.MSELoss()

        self.critic_optim = torch.optim.Adam(self.critic.parameters(),
                                             lr=2e-4,
                                             weight_decay=1e-5)

        self.training = True
        self.eps = 0.2

    @torch.no_grad()
    def take_action(self, s: torch.Tensor, g: torch.Tensor):
        if self.training and np.random.rand() < self.eps:
            return np.random.uniform(-1, 1, self.action_dim)

        act = self.actor(s, g).cpu().detach().squeeze(0).numpy()
        if self.training:
            return self.pertub(act)

        return act

    def pertub(self, act: np.ndarray):
        act += 0.2 * 1.0 * np.random.randn(self.action_dim)
        return act.clip(-1.0, 1.0)
    
    def on_episode_termination(self, sari: Tuple[List[State], List[Action], List[Reward], List[Info]]):
        pass


class HighNetwork(Algorithm):

    def __init__(self, state_dim: int, goal_dim: int, action_dim: int) -> None:
        self.name = 'lesson-high'

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.goal_dim = goal_dim

        self.sac = NewSAC(self.state_dim + self.goal_dim, self.goal_dim, 1.0)

        self.random_episode = 20
        self.reset()

    def reset(self):
        self.dg = None
        self.ag = None
        self.epoch = 0

    def on_env_reset(self, info: Dict[str, Any]):
        self.dg = info['desired_goal']
        self.ag = info['rep_ag']

    @torch.no_grad()
    def take_action(self, s: State) -> ActionInfo:
        assert self.dg is not None
        assert self.ag is not None

        act = None
        if self.epoch <= self.random_episode:
            act = torch.from_numpy(np.random.uniform(-20, 20, self.goal_dim))

        else:
            obs = torch.cat([s, self.dg])
            act = self.sac.take_action(obs)

        return ((self.ag + act).clip(-200, 200), dict(raw_action=act))

    def after_step(self, transition: TransitionTuple[State]):
        # return super().after_step(transition)
        pass

    def on_episode_termination(self, sari: Tuple[List[State], List[Action],
                                                 List[Reward], List[Info]]):
        self.epoch += 1


class LESSON(Algorithm):

    def __init__(self, state_dim: int, goal_dim: int, action_dim: int):
        self.name = "lesson"
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.goal_dim = goal_dim

        self.gamma = 0.99

        self.tau = 5e-3

        self.start_traininig_size = int(1e4)
        self.mini_batch_size = 128

        self.low_network = LowNetwork(self.state_dim, self.goal_dim,
                                      self.action_dim)

        self.high_network = HighNetwork(self.state_dim, self.goal_dim,
                                        self.action_dim)

        self.representation_network = RepresentationNetwork(
            self.state_dim, self.goal_dim)

        self.c = 50
        self.high_random_episode = 20
        self.reset()

    def reset(self):
        self.total_steps = 0
        self.inner_steps = 0
        self.epoch = 0

        self.current_high_act = None
        self.last_high_obs = None
        self.last_high_act = None
        self.high_reward = 0.0

        self.end = False

    def on_env_reset(self, info: Dict[str, Any]):
        assert 'desired_goal' in info
        self.desired_goal = info['desired_goal']

        obs = info['observation']
        info['rep_ag'] = self.representation_network(obs).detach().cpu().nump()

        self.high_network.on_env_reset(info)
        self.low_network.on_env_reset(info)

    def update_high(self):
        raise NotImplementedError()

    @torch.no_grad()
    def take_action(self, state: State) -> Action:
        if self.inner_steps % self.c == 0:
            (act, info) = self.high_network.take_action(state)

            self.current_high_act = act
            self.last_high_obs = state
            self.last_high_act = info['raw_act']

        assert self.current_high_act is not None
        act = self.low_network.take_action(
            state, torch.from_numpy(self.current_high_act))

        return act

    def update_phi(self):
        raise NotImplementedError()

    def after_step(self, transition: TransitionTuple[State]):
        (s1, s2) = transition

        self.high_reward += s1.reward

        if self.inner_steps != 0 and self.inner_steps % (self.c - 1) == 0:
            assert self.last_high_obs is not None
            assert self.last_high_act is not None

            self.high_network.after_step(
                (NotNoneStep(self.last_high_obs, self.last_high_act,
                             self.high_reward), Step(s2.state, None, None)))

            self.high_reward = 0.0

        obs = s2.state
        # self.high_reward += s1.reward

        # self.achieved_goal = self.representation_network(obs).detach().cpu()

        if s1.info['end']:
            self.end = True

        self.inner_steps += 1
        self.total_steps += 1

    def on_episode_termination(self, sari: Tuple[List[State], List[Action],
                                                 List[Reward], List[Info]]):

        self.high_network.on_episode_termination(sari)
        self.low_network.on_episode_termination(sari)
        self.epoch += 1
        self.inner_steps = 0
