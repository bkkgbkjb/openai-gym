import setup
from utils.algorithm import (ActionInfo, Mode, ReportInfo)
from utils.common import Info
from utils.step import NotNoneStep, Step
from utils.transition import (Transition, resolve_transitions)
from torch import nn
from collections import deque
import torch
from utils.preprocess import PreprocessI
from utils.algorithm import Algorithm
from torch.distributions import Categorical, Normal
from typing import Union
from utils.nets import NeuralNetworks, layer_init
from torch.utils.data import DataLoader

from typing import List, Tuple, Any, Optional, Callable, Dict, cast
import numpy as np

from utils.replay_buffer import ReplayBuffer

O = torch.Tensor
Action = torch.Tensor

S = O
Reward = int

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Preprocess(PreprocessI[O, S]):

    def __init__(self):
        pass

    def on_agent_reset(self):
        pass

    def get_current_state(self, h: List[O]) -> S:
        assert len(h) > 0

        # assert h[-1].shape == (4, 1, 84, 84)
        return torch.from_numpy(h[-1]).type(torch.float32).to(DEVICE)


class Actor(NeuralNetworks):

    def __init__(self,
                 state_dim: int,
                 goal_dim: int,
                 action_dim: int,
                 action_scale: float = 1.0):
        super(Actor, self).__init__()
        self.action_scale = action_scale
        self.goal_dim = goal_dim
        self.state_dim = state_dim

        self.net = nn.Sequential(
            layer_init(nn.Linear(self.state_dim + self.goal_dim, 400)),
            nn.ReLU(), layer_init(nn.Linear(400, 300)), nn.ReLU(),
            layer_init(nn.Linear(300, action_dim)), nn.Tanh())

    def forward(self, s: torch.Tensor, g: torch.Tensor):
        return self.action_scale * self.net(torch.cat([s, g], dim=1))


class BCP(Algorithm[S]):

    def __init__(self,
                 state_dim: int,
                 goal_dim: int,
                 action_dim: int,
                 action_scale: float = 1.0):
        self.set_name('bcp')
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.action_scale = action_scale

        self.actor = Actor(self.state_dim, self.goal_dim, self.action_dim,
                           self.action_scale).to(DEVICE)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=3e-3)

        self.actor_loss = nn.MSELoss()

        self.batch_size = 196

        self.reset()

    def reset(self):
        self.times = 0
        self.data_loader = None
        self.replay_buffer = ReplayBuffer(None)
        self.reset_episode_info()

    def on_env_reset(self, mode: Mode, info: Dict[str, Any]):

        assert mode == 'eval'

        assert self.fg is None
        self.fg = torch.from_numpy(info['desired_goal']).type(
            torch.float32).to(DEVICE)

    def reset_episode_info(self):
        self.fg = None

    def on_episode_termination(
        self, mode: Mode, sari: Tuple[List[S], List[Action], List[Reward],
                                      List[Info]]
    ) -> Optional[ReportInfo]:
        (_, _, _, i) = sari
        assert i[-1]['end']
        assert len(i[-1].keys()) == 1
        success = i[-2]['env_info']['is_success']
        self.reset_episode_info()

        return dict(success_rate=1 if success else 0)

    @torch.no_grad()
    def take_action(self, mode: Mode, state: S) -> Union[ActionInfo, Action]:
        assert mode == 'eval'
        s = state.unsqueeze(0).to(DEVICE)

        assert self.fg is not None

        return self.actor(s, self.fg.unsqueeze(0)).squeeze()

    def get_data(self, dataloader: DataLoader):

        for (states, actions, rewards, next_states, dones,
             goals) in dataloader:
            for (s, a, r, sn, done, goal) in zip(states, actions, rewards,
                                                 next_states, dones, goals):
                self.replay_buffer.append(
                    Transition((NotNoneStep(s, a, r.item(),
                                            dict(goal=goal, end=False)),
                                Step(sn, None, None,
                                     dict(end=done.item() == 1)))))

    def manual_train(self, info: Dict[str, Any]):
        assert 'dataloader' in info
        dataloader = info['dataloader']

        if self.data_loader != dataloader:
            assert self.data_loader is None
            self.get_data(dataloader)
            self.data_loader = dataloader

        (states, actions, _, _, _, infos) = resolve_transitions(
            self.replay_buffer.sample(self.batch_size), (self.state_dim, ),
            (self.action_dim, ))
        goals = torch.stack([i['goal'] for i in infos]).to(DEVICE)

        pred_actions = self.actor(states, goals)
        loss = self.actor_loss(pred_actions, actions)

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        self.report(dict(actor_loss=loss))

        return self.batch_size
