import setup
from utils.common import (
    ActionInfo,
    Step,
    Transition,
    NotNoneStep,
)
from torch import nn
from collections import deque
import torch
from utils.nets import NeuralNetworks
from utils.preprocess import PreprocessInterface
from utils.algorithm import AlgorithmInterface

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
Action = np.ndarray

State = Observation
Reward = float

Transition = Transition[State]
Step = Step[State]
NotNoneStep = NotNoneStep[State]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Preprocess(PreprocessInterface[Observation, State]):

    def __init__(self):
        pass

    def on_agent_reset(self):
        pass

    def get_current_state(self, h: List[Observation]) -> State:
        assert len(h) > 0

        # assert h[-1].shape == (4, 1, 84, 84)
        return torch.from_numpy(h[-1]).type(torch.float32).to(DEVICE)


class Actor(NeuralNetworks):

    def __init__(self, n_states: int, n_actions: int):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            layer_init(nn.Linear(n_states, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, n_actions)),
            nn.Tanh(),
        ).to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


class TD3(AlgorithmInterface[State]):

    def __init__(self, n_states: int, n_actions: int) -> None:
        self.name = "td3"
        self.n_actions = n_actions
        self.n_states = n_states

        self.gamma = 0.99
        self.tau = 5e-3

        self.actor = Actor(self.n_states, self.n_actions)
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

        self.replay_buffer = ReplayBuffer[State]((self.n_states, ),
                                                 (self.n_actions, ), int(1e6))

        self.noise_generator = lambda: np.random.normal(
            0, 0.1, size=self.n_actions)

        self.mini_batch_size = 256

        self.times = 0
        self.eval = False
        self.max_action = 1.0
        self.start_timestamp = int(25e3)

    def on_toggle_eval(self, isEval: bool):
        self.eval = isEval

    def set_reporter(self, reporter: Callable[[Dict[str, Any]], None]):
        self.report = reporter

    def train(self):
        (states, actions, rewards, next_states, done) = ReplayBuffer.resolve(
            self.replay_buffer.sample(self.mini_batch_size), (self.n_states, ),
            (self.n_actions, ))

        noise = (torch.distributions.uniform.Uniform(
            -self.max_action * 0.2,
            self.max_action * 0.2).sample(actions.shape).to(DEVICE))

        next_actions = (self.actor_target(next_states) + noise).clamp(
            -self.max_action, self.max_action)

        target_Q1 = self.critic_target1(next_states, next_actions)
        target_Q2 = self.critic_target2(next_states, next_actions)

        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = rewards + (1 - done) * self.gamma * target_Q

        current_Q1 = self.critic1(states, actions)
        current_Q2 = self.critic2(states, actions)
        critic_loss = self.critic1_loss(
            current_Q1, target_Q) + self.critic2_loss(current_Q2, target_Q)

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

            self.report(dict(actor_loss=actor_loss))

    def reset(self):
        self.times = 0
        self.replay_buffer.clear()

    def on_agent_reset(self):
        pass

    @torch.no_grad()
    def take_action(self, state: State) -> Action:
        if self.times <= self.start_timestamp:
            return np.random.uniform(-self.max_action,
                                     self.max_action,
                                     size=self.n_actions)

        act = self.actor(state).cpu()
        if not self.eval:
            noise = self.noise_generator()
            act += torch.from_numpy(noise)
        return act.squeeze(0).numpy().clip(-self.max_action, self.max_action)

    def on_episode_termination(self, sar: Tuple[List[State], List[ActionInfo],
                                                List[Reward]]):
        pass

    def after_step(
        self,
        sar: Tuple[State, ActionInfo, Reward],
        sa: Tuple[State, Optional[ActionInfo]],
    ):
        (s, a, r) = sar
        (sn, an) = sa
        self.replay_buffer.append((s, a, r, sn, an))

        if self.times >= self.start_timestamp:
            self.train()

        self.times += 1
