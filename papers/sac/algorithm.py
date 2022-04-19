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
        assert raw_act.shape == (s.size(0), 6)
        act = torch.tanh(raw_act)

        raw_log_prob = normal.log_prob(raw_act)
        assert raw_log_prob.shape == (s.size(0), 6)

        mod_log_prob = (1 - act.pow(2) + 1e-6).log()
        assert mod_log_prob.shape == (s.size(0), 6)
        log_prob = (raw_log_prob - mod_log_prob).sum(1, keepdim=True)

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

        self.gamma = 0.99

        self.alpha = 1.0
        self.tau = 1e-2

        self.replay_memory_size = int(1e6)
        self.start_traininig_size = int(1e4)
        self.mini_batch_size = 256

        self.online_v = VFunction(self.n_state)
        self.offline_v = VFunction(self.n_state)
        self.offline_v.load_state_dict(self.online_v.state_dict())
        for p in self.offline_v.parameters():
            p.requires_grad = False

        self.offline_v.eval()

        self.policy = PaiFunction(self.n_state, self.n_actions)

        self.q1 = QFunction(self.n_state, self.n_actions)
        self.q2 = QFunction(self.n_state, self.n_actions)

        self.v_loss = nn.MSELoss()
        self.q1_loss = nn.MSELoss()
        self.q2_loss = nn.MSELoss()

        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), 3e-4)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), 3e-4)

        self.v_optimizer = torch.optim.Adam(self.online_v.parameters(), 3e-4)
        self.p_optimizer = torch.optim.Adam(self.policy.parameters(), 3e-4)

        self.reset()

    def set_reporter(self, reporter: Callable[[Dict[str, Any]], None]):
        self.reporter = reporter

    def reset(self):
        self.times = 0
        self.replay_memory: deque[Transition] = deque(
            maxlen=self.replay_memory_size)

    def on_agent_reset(self):
        pass

    @torch.no_grad()
    def take_action(self, state: State) -> Action:
        action, _, _ = self.policy.sample(state.unsqueeze(0))
        return action.detach().cpu().squeeze(0).numpy()

    def on_episode_termination(self, sar: Tuple[List[State], List[ActionInfo[Action]], List[Reward]]):
        pass

    def after_step(self, sar: Tuple[State, ActionInfo[Action], Reward], sa: Tuple[State, Optional[ActionInfo[Action]]]):
        (s, a, r) = sar
        (sn, an) = sa
        self.replay_memory.append((s, a, r, sn, an))

        if len(self.replay_memory) >= self.start_traininig_size:
            self.train()

        self.times += 1

    def get_mini_batch(self) -> List[Transition]:
        idx = np.random.choice(len(self.replay_memory), self.mini_batch_size)
        l = list(self.replay_memory)

        r = []
        for i in idx:
            r.append(l[i])

        return r

    def get_batch_details(self, mini_batch: List[Transition]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        states = torch.stack([s for (s, _, _, _, _) in mini_batch])
        assert states.shape == (self.mini_batch_size, 17)

        actions = torch.stack([torch.from_numpy(a).type(torch.float32)
                               for (_, (a, _), _, _, _) in mini_batch])
        assert actions.shape == (self.mini_batch_size, 6)

        rewards = torch.stack([torch.tensor(r, dtype=torch.float32)
                               for (_, _, r, _, _) in mini_batch]).unsqueeze(1)
        assert rewards.shape == (self.mini_batch_size, 1)

        next_states = torch.stack([sn for (_, _, _, sn, _) in mini_batch])
        assert next_states.shape == (self.mini_batch_size, 17)

        done = torch.as_tensor(
            [1 if an is None else 0 for (_, _, _, _, an) in mini_batch], dtype=torch.int8).unsqueeze(1)
        assert done.shape == (self.mini_batch_size, 1)

        return (states.to(DEVICE), actions.to(DEVICE), rewards.to(DEVICE), next_states.to(DEVICE), done.to(DEVICE))

    def train(self):
        mini_batch = self.get_mini_batch()

        (states, actions, rewards, next_states,
         done) = self.get_batch_details(mini_batch)

        new_actions, new_log_probs, _ = self.policy.sample(states)

        # Training Q Function
        predicted_q1 = self.q1(states, actions)
        predicted_q2 = self.q2(states, actions)
        target_val = self.offline_v(next_states)
        target_q_val = (rewards + (1 - done) *
                        self.gamma * target_val).detach()

        q_val_loss1 = self.q1_loss(predicted_q1, target_q_val)
        q_val_loss2 = self.q2_loss(predicted_q2, target_q_val)

        self.q1_optimizer.zero_grad()
        q_val_loss1.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q_val_loss2.backward()
        self.q2_optimizer.step()

        # Training V Function
        predicted_val = self.online_v(states)
        predicted_new_q_value = torch.min(
            self.q1(states, new_actions), self.q2(states, new_actions))

        target_value_func = (predicted_new_q_value -
                             self.alpha * new_log_probs).detach()

        value_loss = self.v_loss(predicted_val, target_value_func)

        self.v_optimizer.zero_grad()
        value_loss.backward()
        self.v_optimizer.step()

        # Training P Function
        policy_loss = (self.alpha * new_log_probs -
                       predicted_new_q_value).mean()
        self.p_optimizer.zero_grad()
        policy_loss.backward()
        self.p_optimizer.step()

        # soft update old_v net
        for old_params, params in zip(self.offline_v.parameters(), self.online_v.parameters()):
            old_params.data.copy_(
                old_params.data * (1.0 - self.tau) + params.data * self.tau
            )

        self.reporter(dict(
            value_loss=value_loss,
            policy_loss=policy_loss,
            q_loss1=q_val_loss1,
            q_loss2=q_val_loss2
        ))


class Preprocess(PreprocessInterface[Observation, Action, State]):
    def __init__(self):
        pass

    def on_agent_reset(self):
        pass

    def get_current_state(self, h: List[Observation]) -> State:
        assert len(h) > 0

        # assert h[-1].shape == (4, 1, 84, 84)
        return torch.from_numpy(h[-1]).type(torch.float32).to(DEVICE)
