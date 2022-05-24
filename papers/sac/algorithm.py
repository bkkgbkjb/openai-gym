import setup
from utils.common import (
    ActionInfo,
    Transition,
)
from torch import nn
from collections import deque
import torch
from utils.preprocess import Preprocess
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
Reward = int

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class VFunction(NeuralNetworks):

    def __init__(self, n_state: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(n_state, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1)),
        ).to(DEVICE)

    def forward(self, s: State) -> torch.Tensor:
        return self.net(s.to(DEVICE))


class QFunction(NeuralNetworks):

    def __init__(self, n_state: int, n_action: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(n_state + n_action, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1)),
        ).to(DEVICE)

    def forward(self, s: State, a: Action) -> torch.Tensor:
        assert s.size(1) == 17
        assert a.size(1) == 6
        return self.net(torch.cat([s.to(DEVICE), a.to(DEVICE)], 1))


class PaiFunction(NeuralNetworks):

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


class SAC(Algorithm):

    def __init__(self, n_state: int, n_actions: int):
        self.name = "sac"
        self.n_actions = n_actions
        self.n_state = n_state

        self.gamma = 0.99

        self.alpha = 1.0
        self.tau = 1e-2

        self.start_traininig_size = int(1e4)
        self.mini_batch_size = 256

        self.online_v = VFunction(self.n_state)
        self.offline_v = self.online_v.clone().no_grad()

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

    def on_toggle_eval(self, isEval: bool):
        pass

    def set_reporter(self, reporter: Callable[[Dict[str, Any]], None]):
        self.report = reporter

    def reset(self):
        self.times = 0
        self.replay_memory = ReplayBuffer[State]((self.n_state, ),
                                                 (self.n_actions, ))

    def on_agent_reset(self):
        pass

    @torch.no_grad()
    def take_action(self, state: State) -> Action:
        action, _, _ = self.policy.sample(state.unsqueeze(0))
        return action.detach().cpu().squeeze(0).numpy()

    def on_episode_termination(self, sar: Tuple[List[State], List[ActionInfo],
                                                List[Reward]]):
        pass

    def after_step(self, transition: Transition):
        (s, a, r, sn, an) = transition
        assert isinstance(an, tuple) or an is None
        self.replay_memory.append((s, a, r, sn, an))

        if self.replay_memory.len >= self.start_traininig_size:
            self.train()

        self.times += 1

    def train(self):

        (states, actions, rewards, next_states, done) = ReplayBuffer.resolve(
            self.replay_memory.sample(self.mini_batch_size), (self.n_state, ),
            (self.n_actions, ))

        new_actions, new_log_probs, _ = self.policy.sample(states)

        # Training Q Function
        predicted_q1 = self.q1(states, actions)
        predicted_q2 = self.q2(states, actions)
        target_val = self.offline_v(next_states)
        target_q_val = (rewards +
                        (1 - done) * self.gamma * target_val).detach()

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
        predicted_new_q_value = torch.min(self.q1(states, new_actions),
                                          self.q2(states, new_actions))

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
        self.offline_v.soft_update_to(self.online_v, self.tau)

        self.report(
            dict(
                value_loss=value_loss,
                policy_loss=policy_loss,
                q_loss1=q_val_loss1,
                q_loss2=q_val_loss2,
            ))


class Preprocess(Preprocess[Observation]):

    def __init__(self):
        pass

    def on_agent_reset(self):
        pass

    def get_current_state(self, h: List[Observation]) -> State:
        assert len(h) > 0

        # assert h[-1].shape == (4, 1, 84, 84)
        return torch.from_numpy(h[-1]).type(torch.float32).to(DEVICE)
