import setup
from utils.common import ActionInfo, Transition, TransitionTuple, resolve_transitions
from torch import nn
import math
from collections import deque
import torch
from utils.env_sb3 import LazyFrames, resolve_lazy_frames
from utils.preprocess import Preprocess
from utils.algorithm import Algorithm
from typing import List, Tuple, Optional, cast, Callable, Dict, Any
import numpy as np
from utils.nets import NeuralNetworks, layer_init
from utils.replay_buffer import ReplayBuffer

Observation = LazyFrames
Action = np.ndarray

State = Observation
Reward = int

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Preprocess(Preprocess[Observation]):

    def __init__(self):
        pass

    def on_agent_reset(self):
        pass

    def get_current_state(self, h: List[Observation]) -> State:
        assert len(h) > 0

        assert h[-1].shape == (4, 84, 84)
        return h[-1]


class QNetwork(NeuralNetworks):

    def __init__(self, n_actions: int):
        super(QNetwork, self).__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, (8, 8), 4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, (4, 4), 2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, (3, 3), 1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(7 * 7 * 64, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, n_actions)),
        ).to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rlt = self.net(x.to(DEVICE))
        assert rlt.shape == (x.shape[0], self.n_actions)
        return rlt


class DQNAlgorithm(Algorithm):

    def __init__(self, n_actions: int, gamma: float = 0.99):
        self.name = "dqn"
        self.n_actions = n_actions

        self.times = 0

        self.online_network = QNetwork(n_actions)
        self.optimizer = torch.optim.Adam(self.online_network.parameters(),
                                          lr=1e-4)

        self.target_network = self.online_network.clone().no_grad()

        self.batch_size = 32

        self.update_target = 250

        self.replay_memory = ReplayBuffer((4, 84, 84), (1, ), int(2e5))

        self.gamma = gamma
        self.loss_func = torch.nn.MSELoss()

        self.update_times = 4
        self.eval = False

    def on_toggle_eval(self, isEval: bool):
        self.eval = isEval

    @torch.no_grad()
    def take_action(self, state: State) -> Action:
        if self.eval:
            act_vals = self.online_network(
                resolve_lazy_frames(state).unsqueeze(0))
            maxi = torch.argmax(act_vals)
            return np.asarray([maxi.item()], dtype=np.int64)

        rand = np.random.random()
        max_decry_times = 100_0000
        sigma = 1 - 0.95 / max_decry_times * np.min(
            [self.times, max_decry_times])

        if rand < sigma:
            return np.asarray([np.random.choice(self.n_actions)],
                              dtype=np.int64)

        else:
            act_vals = self.online_network(
                resolve_lazy_frames(state).unsqueeze(0))
            maxi = torch.argmax(act_vals)
            return np.asarray([maxi.item()], dtype=np.int64)

    def after_step(self, transition: TransitionTuple):
        (s, a, r, sn, an) = transition
        assert isinstance(an, tuple) or an is None
        self.replay_memory.append(Transition((s, a, r, sn, an)))

        if self.times != 0 and self.times % (self.update_times) == 0:

            if self.replay_memory.len >= 5 * self.batch_size:

                self.train()

        if (self.times != 0 and self.times %
            (self.update_target * self.update_times) == 0):
            self.update_target_network()

        self.times += 1

    def update_target_network(self):
        self.target_network.hard_update_to(self.online_network)

    def train(self):

        (states, actions, rewards, next_states, done) = resolve_transitions(
            self.replay_memory.sample(self.batch_size), (4, 84, 84), (1, ))

        q_next = self.target_network(next_states)

        max_values = (torch.max(
            q_next,
            dim=1,
        )[0].unsqueeze(1))

        assert max_values.shape == (32, 1)

        target = rewards + (1 - done) * self.gamma * max_values

        assert target.shape == (32, 1)

        x_vals = self.online_network(states)

        x = x_vals.gather(1, actions)

        assert x.shape == (32, 1)

        loss = self.loss_func(x, target)

        self.optimizer.zero_grad()
        loss.backward()

        for param in self.online_network.parameters():  # gradient clipping
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        self.report(dict(loss=loss))

    def on_episode_termination(self, sar: Tuple[List[State], List[ActionInfo],
                                                List[Reward]]):
        (s, a, r) = sar
        assert len(s) == len(a) + 1
        assert len(s) == len(r) + 1
        pass


class DDQNAlgorithm(DQNAlgorithm, Algorithm):

    def __init__(self, n_actions: int, gamma: float = 0.99):
        super().__init__(n_actions, gamma)
        self.update_target = 1250
        self.name = "ddqn"

    def train(self):
        (states, actions, rewards, next_states, done) = resolve_transitions(
            self.replay_memory.sample(self.batch_size), (4, 84, 84), (1, ))

        q_next = self.target_network(next_states)

        max_values = q_next.gather(
            1,
            torch.argmax(self.online_network(next_states).detach(),
                         dim=1,
                         keepdim=True)).squeeze(1)

        assert max_values.shape == (32, 1)

        target = rewards + (1 - done) * self.gamma * max_values

        assert target.shape == (32, 1)

        x_vals = self.online_network(states)

        x = x_vals.gather(1, actions)

        assert x.shape == (32, 1)

        loss = self.loss_func(x, target)

        self.optimizer.zero_grad()
        loss.backward()

        for param in self.online_network.parameters():  # gradient clipping
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        self.report(dict(loss=loss))
