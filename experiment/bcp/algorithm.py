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
                                                lr=5e-3)

        self.actor_optim_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.actor_optimizer,
            patience=500,
            factor=0.25,
            verbose=True,
            threshold=1e-4,
            cooldown=100)

        self.actor_loss = nn.MSELoss()

        self.batch_size = 196

        self.reset()

    def reset(self):
        self.times = 0
        self.dataset = None
        self.replay_buffer = ReplayBuffer(None)
        self.reset_episode_info()

    def on_env_reset(self, mode: Mode, info: Dict[str, Any]):

        assert mode == 'eval'

        assert self.fg is None
        self.fg = torch.from_numpy(info['desired_goal']).type(
            torch.float32).to(DEVICE)

        assert self.env_goal_type is None
        self.env_goal_type = info['env'].env_info['goal_type']

    def reset_episode_info(self):
        self.fg = None
        self.env_goal_type = None

    def on_episode_termination(
        self, mode: Mode, sari: Tuple[List[S], List[Action], List[Reward],
                                      List[Info]]
    ) -> Optional[ReportInfo]:
        (_, _, _, i) = sari
        assert i[-1]['end']
        assert len(i[-1].keys()) == 1
        success = i[-2]['env_info']['is_success']

        report = {f'{self.env_goal_type}_success_rate': 1 if success else 0}

        self.reset_episode_info()
        return report

    @torch.no_grad()
    def take_action(self, mode: Mode, state: S) -> Union[ActionInfo, Action]:
        assert mode == 'eval'
        s = state.unsqueeze(0).to(DEVICE)

        assert self.fg is not None

        return self.actor(s, self.fg.unsqueeze(0)).squeeze()

    def get_data(self, dataset: Any):

        states = torch.from_numpy(dataset['state'][:]).type(torch.float32)
        actions = torch.from_numpy(dataset['action'][:]).type(torch.float32)
        rewards = torch.from_numpy(dataset['reward'][:]).type(torch.float32)
        next_states = torch.from_numpy(dataset['next_state'][:]).type(
            torch.float32)
        dones = torch.from_numpy(dataset['done'][:]).type(torch.float32)
        goals = torch.from_numpy(dataset['info']['goal'][:]).type(
            torch.float32)

        assert len(states) == len(actions) == len(rewards) == len(
            next_states) == len(dones) == len(goals)

        for i in range(len(states)):
            s = states[i]
            a = actions[i]
            r = rewards[i]
            sn = next_states[i]
            d = dones[i]
            g = goals[i]

            self.replay_buffer.append(
                Transition((NotNoneStep(s, a, r.item(), dict(goal=g,
                                                             end=False)),
                            Step(sn, None, None, dict(end=d.item() == 1)))))

    def manual_train(self, info: Dict[str, Any]):
        assert 'dataset' in info
        dataset = info['dataset']
        if self.dataset != dataset:
            assert self.dataset is None
            self.get_data(dataset)
            self.dataset = dataset

        (states, actions, _, _, _, infos) = resolve_transitions(
            self.replay_buffer.sample(self.batch_size), (self.state_dim, ),
            (self.action_dim, ))
        goals = torch.stack([i['goal'] for i in infos]).to(DEVICE)

        pred_actions = self.actor(states, goals)
        loss = self.actor_loss(pred_actions, actions)

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        self.actor_optim_scheduler.step(loss.item())

        self.report(dict(actor_loss=loss))

        return self.batch_size
