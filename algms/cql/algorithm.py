import setup
from utils.transition import (Transition, TransitionTuple, resolve_transitions)
from utils.algorithm import ActionInfo, Mode
from torch import nn
from collections import deque
import torch
from utils.preprocess import PreprocessI
from utils.step import (NotNoneStep, Step)
from utils.algorithm import Algorithm
from torch.distributions import Categorical, Normal
from typing import Union
from utils.nets import NeuralNetworks, layer_init
from torch.utils.data import DataLoader

from typing import List, Tuple, Any, Optional, Callable, Dict, cast
import numpy as np

from utils.replay_buffer import ReplayBuffer

Observation = torch.Tensor
Action = torch.Tensor

State = Observation
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


class CQL_SAC(Algorithm[State]):

    def __init__(self, n_state: int, n_actions: int):
        self.name = "cql-sac"
        self.n_actions = n_actions
        self.n_state = n_state

        self.gamma = 0.99

        self.tau = 5e-3

        self.start_traininig_size = int(1e4)

        self.policy = PaiFunction(self.n_state, self.n_actions)

        self.q1 = QFunction(self.n_state, self.n_actions)
        self.q2 = QFunction(self.n_state, self.n_actions)

        self.q1_target = self.q1.clone().no_grad()
        self.q2_target = self.q2.clone().no_grad()

        self.q1_loss = nn.MSELoss()
        self.q2_loss = nn.MSELoss()

        self.log_alpha = torch.tensor([0.1], requires_grad=True, device=DEVICE)
        assert self.log_alpha.requires_grad
        self.target_entropy = -n_actions

        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), 3e-4)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), 3e-4)

        self.log_alpha_optimizer = torch.optim.Adam(params=[self.log_alpha],
                                                    lr=3e-4)

        self.p_optimizer = torch.optim.Adam(self.policy.parameters(), 3e-4)

        self.num_repeat_actions = 10
        self.temperature = 1.0
        self.cql_weight = 1.0
        self.mini_batch_size = 256

        self.reset()

    def reset(self):
        self.times = 0
        self.transitions = None
        self.replay_memory = ReplayBuffer(None)

    def get_data(self, transitions: List[Transition]):
        for transition in transitions:
            self.replay_memory.append(transition)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.no_grad()
    def take_action(self, mode: Mode, state: State) -> Action:
        assert mode == 'eval'
        action, _, max_acts = self.policy.sample(state.unsqueeze(0))
        return (max_acts if mode == 'eval' else action).squeeze(0)

    def after_step(self, _: Mode, transition: TransitionTuple[State]):
        self.times += 1

    def manual_train(self, info: Dict[str, Any]):
        assert "transitions" in info
        transitions: List[Transition] = info['transitions']
        if self.transitions != transitions:
            self.get_data(transitions)
            self.transitions = transitions

        self.train()
        return self.mini_batch_size

    def calc_pi_values(self, states: torch.Tensor, pred_states: torch.Tensor):
        acts, log_pis, _ = self.policy.sample(states)

        q1 = self.q1(pred_states, acts)
        q2 = self.q2(pred_states, acts)

        return q1 - log_pis.detach(), q2 - log_pis.detach()

    def calc_random_values(self, states: torch.Tensor, actions: torch.Tensor):
        random_value1 = self.q1(states, actions)
        random_log_prob1 = np.log(0.5**actions.shape[-1])

        random_value2 = self.q2(states, actions)
        random_log_prob2 = np.log(0.5**actions.shape[-1])

        return random_value1 - random_log_prob1, random_value2 - random_log_prob2

    def train(self):

        (states, actions, rewards, next_states, done, _) = resolve_transitions(
            self.replay_memory.sample(self.mini_batch_size), (self.n_state, ),
            (self.n_actions, ))

        # Training P Function
        new_actions, new_log_probs, _ = self.policy.sample(states)
        policy_loss = (self.alpha.detach() * new_log_probs -
                       torch.min(self.q1(states, new_actions),
                                 self.q2(states, new_actions))).mean()

        self.p_optimizer.zero_grad()
        policy_loss.backward()
        self.p_optimizer.step()

        # Training self.alpha
        alpha_loss = -(self.alpha *
                       (new_log_probs + self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # Training Q Function
        next_actions, next_actions_log_probs, _ = self.policy.sample(
            next_states)

        q1_target = self.q1_target(next_states, next_actions)
        q2_target = self.q2_target(next_states, next_actions)

        q_target_next = torch.min(
            q1_target,
            q2_target) - self.alpha.detach() * next_actions_log_probs

        target_q_val = (rewards + self.gamma *
                        (1 - done) * q_target_next).detach()

        predicted_q1 = self.q1(states, actions)
        predicted_q2 = self.q2(states, actions)

        q_val_loss1 = self.q1_loss(predicted_q1, target_q_val)
        q_val_loss2 = self.q2_loss(predicted_q2, target_q_val)

        # CQL addon
        random_actions = torch.FloatTensor(
            self.mini_batch_size * self.num_repeat_actions,
            actions.shape[-1]).uniform_(-1, 1).to(DEVICE)

        obs_len = len(states.shape)

        repeat_size = [1, self.num_repeat_actions] + [1] * (obs_len - 1)
        view_size = [self.mini_batch_size * self.num_repeat_actions] + list(
            states.shape[1:])

        tmp_obs = states.unsqueeze(1).repeat(*repeat_size).view(*view_size)
        tmp_obs_next = next_states.unsqueeze(1).repeat(*repeat_size).view(
            *view_size)

        current_pi_value1, current_pi_value2 = self.calc_pi_values(
            tmp_obs, tmp_obs)
        next_pi_value1, next_pi_value2 = self.calc_pi_values(
            tmp_obs_next, tmp_obs)

        random_value1, random_value2 = self.calc_random_values(
            tmp_obs, random_actions)

        current_pi_value1 = current_pi_value1.reshape(self.mini_batch_size,
                                                      self.num_repeat_actions,
                                                      1)
        current_pi_value2 = current_pi_value2.reshape(self.mini_batch_size,
                                                      self.num_repeat_actions,
                                                      1)

        next_pi_value1 = next_pi_value1.reshape(self.mini_batch_size,
                                                self.num_repeat_actions, 1)
        next_pi_value2 = next_pi_value2.reshape(self.mini_batch_size,
                                                self.num_repeat_actions, 1)

        random_value1 = random_value1.reshape(self.mini_batch_size,
                                              self.num_repeat_actions, 1)

        random_value2 = random_value2.reshape(self.mini_batch_size,
                                              self.num_repeat_actions, 1)

        # cat q values
        cat_q1 = torch.cat([random_value1, current_pi_value1, next_pi_value1],
                           1)
        cat_q2 = torch.cat([random_value2, current_pi_value2, next_pi_value2],
                           1)
        # shape: (batch_size, 3 * num_repeat, 1)

        assert cat_q1.shape == (self.mini_batch_size,
                                3 * self.num_repeat_actions, 1)
        assert cat_q2.shape == (self.mini_batch_size,
                                3 * self.num_repeat_actions, 1)

        cql1_scaled_loss = \
            (torch.logsumexp(cat_q1 / self.temperature, dim=1).mean() * \
            self.cql_weight * self.temperature - predicted_q1.mean()) * \
            self.cql_weight
        cql2_scaled_loss = \
            (torch.logsumexp(cat_q2 / self.temperature, dim=1).mean() * \
            self.cql_weight * self.temperature - predicted_q2.mean()) * \
            self.cql_weight

        self.q1_optimizer.zero_grad()
        (q_val_loss1 + cql1_scaled_loss).backward(retain_graph=True)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        (q_val_loss2 + cql2_scaled_loss).backward()
        self.q2_optimizer.step()

        self.q1_target.soft_update_to(self.q1, self.tau)
        self.q2_target.soft_update_to(self.q2, self.tau)

        self.report(
            dict(
                # value_loss=value_loss,
                alpha=self.alpha,
                alpha_loss=alpha_loss,
                policy_loss=policy_loss,
                q_loss1=q_val_loss1,
                q_loss2=q_val_loss2,
            ))
