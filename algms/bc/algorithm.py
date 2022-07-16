import setup
from utils.algorithm import (ActionInfo, Mode)
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
                 action_dim: int,
                 action_scale: float = 1.0):
        super(Actor, self).__init__()
        self.net = nn.Sequential(layer_init(nn.Linear(state_dim, 400)),
                                 nn.ReLU(), layer_init(nn.Linear(400, 300)),
                                 nn.ReLU(),
                                 layer_init(nn.Linear(300, action_dim)),
                                 nn.Tanh())
        self.action_scale = action_scale

    def forward(self, s: torch.Tensor):
        return self.action_scale * self.net(s)


class BC(Algorithm[S]):

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 action_scale: float = 1.0):
        self.set_name('bc')
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_scale = action_scale

        self.actor = Actor(self.state_dim, self.action_dim,
                           self.action_scale).to(DEVICE)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=3e-4)

        self.actor_loss = nn.MSELoss()

        self.batch_size = 196

        self.reset()

    def reset(self):
        self.times = 0
        self.data_loader = None
        self.replay_buffer = ReplayBuffer(None)

    @torch.no_grad()
    def take_action(self, mode: Mode, state: S) -> Union[ActionInfo, Action]:
        s = state.unsqueeze(0).to(DEVICE)

        return self.actor(s).squeeze()

    def get_data(self, dataloader: DataLoader):

        for (states, actions, rewards, next_states, dones) in dataloader:
            for (s, a, r, sn, done) in zip(states, actions, rewards,
                                           next_states, dones):
                self.replay_buffer.append(
                    Transition((NotNoneStep(s, a, r.item()),
                                Step(sn, None, None,
                                     dict(end=done.item() == 1)))))

    def manual_train(self, info: Dict[str, Any]):
        assert 'dataloader' in info
        dataloader = info['dataloader']

        if self.data_loader != dataloader:
            self.get_data(dataloader)
            self.data_loader = dataloader

        (states, actions, _, _, _,
         _) = resolve_transitions(self.replay_buffer.sample(self.batch_size),
                                  (self.state_dim, ), (self.action_dim, ))

        pred_actions = self.actor(states)
        loss = self.actor_loss(pred_actions, actions)

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        self.report(dict(actor_loss=loss))

        return self.batch_size
