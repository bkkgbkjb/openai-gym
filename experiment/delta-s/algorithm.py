import setup
from utils.algorithm import (ActionInfo, Mode, ReportInfo)
from utils.common import Info
from utils.episode import Episodes
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


class LowActor(NeuralNetworks):

    def __init__(self,
                 state_dim: int,
                 delta_state_dim: int,
                 action_dim: int,
                 action_scale: float = 1.0):
        super(LowActor, self).__init__()
        self.action_scale = action_scale
        self.delta_state_dim = delta_state_dim
        self.state_dim = state_dim

        self.net = nn.Sequential(
            layer_init(nn.Linear(self.state_dim + self.delta_state_dim, 400)),
            nn.ReLU(), layer_init(nn.Linear(400, 300)), nn.ReLU(),
            layer_init(nn.Linear(300, action_dim)), nn.Tanh())

    def forward(self, s: torch.Tensor, g: torch.Tensor):
        return self.action_scale * self.net(torch.cat([s, g], dim=1))


class HighActor(NeuralNetworks):

    def __init__(self,
                 state_dim: int,
                 goal_dim: int,
                 delta_state_dim: int,
                 action_scale: float = 1.0):
        super(HighActor, self).__init__()
        self.action_scale = action_scale
        self.goal_dim = goal_dim
        self.state_dim = state_dim

        self.net = nn.Sequential(
            layer_init(nn.Linear(self.state_dim + self.goal_dim, 400)),
            nn.ReLU(), layer_init(nn.Linear(400, 300)), nn.ReLU(),
            layer_init(nn.Linear(300, delta_state_dim)), nn.Tanh())

    def forward(self, s: torch.Tensor, g: torch.Tensor):
        return self.action_scale * self.net(torch.cat([s, g], dim=1))


class DeltaS(Algorithm[S]):

    def __init__(self,
                 state_dim: int,
                 goal_dim: int,
                 action_dim: int,
                 action_scale: float = 1.0):
        self.set_name('delta-s')
        self.state_dim = self.delta_state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.action_scale = action_scale

        self.high_actor = HighActor(self.state_dim, self.goal_dim,
                                    self.delta_state_dim, 30).to(DEVICE)
        self.high_actor_optim = torch.optim.Adam(self.high_actor.parameters(),
                                                 lr=5e-3)
        self.high_actor_optim_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.high_actor_optim,
            patience=750,
            factor=0.25,
            min_lr=1e-8,
            cooldown=150,
            verbose=True,
        )

        self.low_actor = LowActor(self.state_dim, self.delta_state_dim,
                                  self.action_dim,
                                  self.action_scale).to(DEVICE)

        self.low_actor_optim = torch.optim.Adam(self.low_actor.parameters(),
                                                lr=5e-3)

        self.low_actor_optim_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.low_actor_optim,
            patience=750,
            factor=0.25,
            verbose=True,
            min_lr=1e-8,
            cooldown=150)

        self.actor_loss = nn.MSELoss()

        self.batch_size = 196
        self.c = 15

        self.reset()

    def reset(self):
        self.times = 0
        self.dataset = None
        self.replay_buffer = ReplayBuffer[Episodes](None)
        self.reset_episode_info()

    def reset_episode_info(self):
        self.fg = None
        self.env_goal_type = None

    def on_env_reset(self, mode: Mode, info: Dict[str, Any]):

        assert mode == 'eval'

        assert self.fg is None
        self.fg = torch.from_numpy(info['desired_goal']).type(
            torch.float32).to(DEVICE)

        assert self.env_goal_type is None
        self.env_goal_type = info['env'].env_info['goal_type']

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
        assert self.fg is not None

        s = state.unsqueeze(0).to(DEVICE)

        delta_s = self.high_actor(s, self.fg.unsqueeze(0))

        return self.low_actor(s, delta_s).squeeze()

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

        s = []
        a = []
        r = []
        info = []

        g = None

        cnt = 0

        for i in range(len(states)):
            s.append(states[i])
            a.append(actions[i])
            r.append(rewards[i].item())
            info.append(dict(goal=goals[i], end=False))

            if g is None:
                g = goals[i]
            else:
                assert torch.equal(g, goals[i])

            if dones[i]:
                if len(s) <= 15:
                    s = []
                    a = []
                    r = []
                    info = []
                    g = None
                    cnt += 1
                    continue

                s.append(next_states[i])
                info.append(dict(end=True))

                self.replay_buffer.append(Episodes.from_list((s, a, r, info)))
                s = []
                a = []
                r = []
                info = []
                g = None

        print(f'short episode cnt is: {cnt}')

    def sample_data(self, episodes: ReplayBuffer[Episodes]):
        episodes_sampled = episodes.sample(self.batch_size)

        data = [[(ss[0].state, ss[0].action, ss[-1].state - ss[0].state,
                  ss[0].info['goal']) for ss in e.cut(self.c)]
                for e in episodes_sampled]

        states = torch.stack([state for e in data for (state, _, _, _) in e]).to(DEVICE)
        diffs = torch.stack([diff for e in data for (_, _, diff, _) in e]).to(DEVICE)
        assert states.shape[1] == diffs.shape[1] == self.state_dim

        actions = torch.stack(
            [action for e in data for (_, action, _, _) in e]).to(DEVICE)

        assert actions.shape[1] == self.action_dim

        goals = torch.stack([goal for e in data for (_, _, _, goal) in e]).to(DEVICE)

        assert goals.shape[1] == self.goal_dim

        assert states.shape[0] == actions.shape[0] == diffs.shape[
            0] == goals.shape[0]

        return states, actions, diffs, goals

    def manual_train(self, info: Dict[str, Any]):
        assert 'dataset' in info
        dataset = info['dataset']
        if self.dataset != dataset:
            assert self.dataset is None
            self.get_data(dataset)
            self.dataset = dataset

        (states, actions, diffs, goals) = self.sample_data(self.replay_buffer)

        high_pred_actions = self.high_actor(states, goals)
        high_loss = self.actor_loss(high_pred_actions, diffs)

        self.high_actor_optim.zero_grad()
        high_loss.backward()
        self.high_actor_optim.step()

        self.high_actor_optim_lr.step(high_loss.item())

        low_pred_actions = self.low_actor(states, diffs)
        low_loss = self.actor_loss(low_pred_actions, actions)

        self.low_actor_optim.zero_grad()
        low_loss.backward()
        self.low_actor_optim.step()

        self.low_actor_optim_lr.step(low_loss.item())

        self.report(dict(low_loss=low_loss, high_loss=high_loss))

        return self.batch_size
