import setup
from utils.common import Info
from utils.transition import (Transition, TransitionTuple, resolve_transitions)
from torch import nn
import math
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
from utils.nets import layer_init
import numpy as np

from utils.replay_buffer import ReplayBuffer

Observation = torch.Tensor
Action = np.ndarray

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

    def __init__(self, n_states: int, n_actions: int) -> None:
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            layer_init(nn.Linear(n_states, 400)),
            nn.LayerNorm(400),
            nn.ReLU(),
            layer_init(nn.Linear(400, 300)),
            nn.LayerNorm(300),
            nn.ReLU(),
            layer_init(nn.Linear(300, n_actions)),
            nn.Tanh(),
        ).to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(NeuralNetworks):

    def __init__(self, n_states: int, n_actions: int) -> None:
        super(Critic, self).__init__()
        self.net1 = nn.Sequential(
            layer_init(nn.Linear(n_states, 400)),
            nn.LayerNorm(400),
            nn.ReLU(),
        ).to(DEVICE)

        self.net2 = nn.Sequential(
            layer_init(nn.Linear(400 + n_actions, 300)),
            nn.LayerNorm(300),
            nn.ReLU(),
            layer_init(nn.Linear(300, 1)),
        ).to(DEVICE)

    def forward(self, state: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        state_value = self.net1(state)
        return self.net2(torch.cat([state_value, action], 1))


class OrnsteinUhlenbeckActionNoise:

    def __init__(self, mu, sigma, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def noise(self):
        x = (self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt +
             self.sigma * np.sqrt(self.dt) *
             np.random.normal(size=self.mu.shape))
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(
            self.mu)
        return self


class DDPG(Algorithm[State]):

    def __init__(self, n_states: int, n_actions: int):
        self.name = "ddpg"
        self.n_actions = n_actions
        self.n_states = n_states

        self.gamma = 0.99
        self.tau = 1e-2

        self.actor = Actor(self.n_states, self.n_actions)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=1e-4)
        self.actor_loss = nn.MSELoss()

        self.actor_target = self.actor.clone().no_grad()

        self.critic = Critic(self.n_states, self.n_actions)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=1e-3,
                                                 weight_decay=1e-2)
        self.critic_loss = nn.MSELoss()

        self.critic_target = self.critic.clone().no_grad()

        self.replay_buffer = ReplayBuffer(int(1e6))

        self.noise_generator = OrnsteinUhlenbeckActionNoise(
            np.zeros(n_actions), sigma=0.2 * np.ones(n_actions)).reset()

        self.mini_batch_size = 64

        self.start_train_ratio = 0.1

        self.times = 0
        self.eval = False

    def on_toggle_eval(self, isEval: bool):
        self.eval = isEval

    def train(self):
        (states, actions, rewards, next_states, done) = resolve_transitions(
            self.replay_buffer.sample(self.mini_batch_size), (self.n_states, ),
            (self.n_actions, ))

        target_q_value = rewards + self.gamma * (
            1 - done) * self.critic_target(next_states,
                                           self.actor_target(next_states))
        current_q_value = self.critic(states, actions)

        self.critic_optimizer.zero_grad()
        value_loss = self.critic_loss(current_q_value, target_q_value)
        value_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss = (-self.critic(states, self.actor(states))).mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.actor_target.soft_update_to(self.actor, self.tau)
        self.critic_target.soft_update_to(self.critic, self.tau)

        self.report(dict(policy_loss=policy_loss, value_loss=value_loss))

    def reset(self):
        self.times = 0
        self.replay_buffer.clear()
        self.noise_generator.reset()

    @torch.no_grad()
    def take_action(self, state: State) -> Action:
        self.actor.eval()
        act = self.actor(state)
        self.actor.train()
        if not self.eval:
            noise = self.noise_generator.noise()
            act += torch.from_numpy(noise).to(DEVICE)
        return act.cpu().squeeze(0).numpy()

    def on_episode_termination(self, sari: Tuple[List[State], List[Action],
                                                 List[Reward], List[Info]]):
        self.noise_generator.reset()

    def after_step(self, transition: TransitionTuple[State]):
        self.replay_buffer.append(Transition(transition))

        assert self.replay_buffer.size is not None

        if self.replay_buffer.len >= math.ceil(
                self.start_train_ratio * self.replay_buffer.size):
            self.train()

        self.times += 1
