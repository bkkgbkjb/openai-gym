import setup
from utils.transition import (
    Transition,
    TransitionTuple,
    resolve_transitions,
)
from utils.common import Action
from torch import nn
from collections import deque
import torch
from utils.nets import NeuralNetworks
from utils.preprocess import PreprocessI
from utils.algorithm import Algorithm

from typing import (
    List,
    Tuple,
    Any,
    Optional,
    Callable,
    Dict,
)
from utils.replay_buffer import ReplayBuffer
from utils.nets import layer_init
import numpy as np

Observation = torch.Tensor

State = Observation
Reward = float

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


class Actor(NeuralNetworks):

    def __init__(self, n_states: int, n_actions: int, action_scale: float):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            layer_init(nn.Linear(n_states, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, n_actions)),
            nn.Tanh(),
        ).to(DEVICE)

        self.action_scale = action_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.action_scale * self.net(x)


class Critic(NeuralNetworks):

    def __init__(self, n_states: int, n_actions: int):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(n_states + n_actions, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1)),
        ).to(DEVICE)

    def forward(self, state: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], 1))


class TD3(Algorithm[State]):

    def __init__(self, n_states: int, n_actions: int,
                 action_scale: float) -> None:
        self.set_name('td3')
        self.n_actions = n_actions
        self.n_states = n_states

        self.gamma = 0.99
        self.tau = 5e-3

        self.action_scale = action_scale

        self.actor = Actor(self.n_states, self.n_actions, self.action_scale)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=3e-4)
        self.actor_loss = nn.MSELoss()

        self.actor_target = self.actor.clone().no_grad()

        self.critic1 = Critic(self.n_states, self.n_actions)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(),
                                                  lr=3e-4)
        self.critic1_loss = nn.MSELoss()

        self.critic_target1 = self.critic1.clone().no_grad()

        self.critic2 = Critic(self.n_states, self.n_actions)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(),
                                                  lr=3e-4)
        self.critic2_loss = nn.MSELoss()

        self.critic_target2 = self.critic2.clone().no_grad()

        self.replay_buffer = ReplayBuffer(int(1e6))

        self.noise_generator = lambda: np.random.normal(
            0, 0.1 * self.action_scale, size=self.n_actions)

        self.mini_batch_size = 256

        self.times = 0
        self.eval = False
        self.start_timestamp = int(10e3)

    def on_toggle_eval(self, isEval: bool):
        self.eval = isEval

    def train(self):
        (states, actions, rewards, next_states, done) = resolve_transitions(
            self.replay_buffer.sample(self.mini_batch_size), (self.n_states, ),
            (self.n_actions, ))

        noise = (torch.randn_like(actions) * 0.2 * self.action_scale).clamp(
            -self.action_scale / 2, self.action_scale / 2)

        next_actions = (self.actor_target(next_states) + noise).clamp(
            -self.action_scale, self.action_scale)

        target_Q1 = self.critic_target1(next_states, next_actions)
        target_Q2 = self.critic_target2(next_states, next_actions)

        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = rewards + (1 - done) * self.gamma * target_Q

        current_Q1 = self.critic1(states, actions)
        current_Q2 = self.critic2(states, actions)
        critic_loss = (self.critic1_loss(current_Q1, target_Q) +
                       self.critic2_loss(current_Q2, target_Q)) / 2

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        self.report(dict(critic_loss=critic_loss))

        if self.times % 2 == 0:
            actor_loss = -self.critic1(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_target1.soft_update_to(self.critic1, self.tau)
            self.critic_target2.soft_update_to(self.critic2, self.tau)

            self.actor_target.soft_update_to(self.actor, self.tau)

            self.report(dict(actor_loss=actor_loss))

    def reset(self):
        self.times = 0
        self.replay_buffer.clear()

    @torch.no_grad()
    def take_action(self, state: State) -> Action:
        if self.times <= self.start_timestamp:
            return torch.from_numpy( np.random.uniform(-self.action_scale,
                                     self.action_scale,
                                     size=self.n_actions)).type(torch.float32)

        act = self.actor(state.unsqueeze(0)).cpu()
        if not self.eval:
            noise = torch.from_numpy( self.noise_generator()).type(torch.float32)
            act += noise
        return act.squeeze(0).clip(-self.action_scale, self.action_scale)

    def after_step(self, transition: TransitionTuple[State]):
        self.replay_buffer.append(Transition(transition))

        if self.times >= self.start_timestamp:
            self.train()

        self.times += 1
