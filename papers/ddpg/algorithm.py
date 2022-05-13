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
from utils.nets import NeuralNetworks
from utils.preprocess import PreprocessInterface
from utils.algorithm import AlgorithmInterface
import plotly.graph_objects as go
from torch.distributions import Categorical, Normal
from tqdm.autonotebook import tqdm

from torchvision import transforms as T
from utils.agent import Agent
from gym.spaces import Box
from typing import (
    Deque,
    List,
    Tuple,
    Literal,
    Any,
    Optional,
    cast,
    Callable,
    Union,
    Iterable,
    Dict,
)
import gym
import numpy.typing as npt
from utils.env import PreprocessObservation, FrameStack, resolve_lazy_frames
import numpy as np

Observation = torch.Tensor
Action = torch.Tensor

State = Observation
Reward = float

Transition = TransitionGeneric[State, Action]
Step = StepGeneric[State, ActionInfo[Action]]
NotNoneStep = NotNoneStepGeneric[State, ActionInfo[Action]]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Preprocess(PreprocessInterface[Observation, Action, State]):
    def __init__(self):
        pass

    def on_agent_reset(self):
        pass

    def get_current_state(self, h: List[Observation]) -> State:
        assert len(h) > 0

        # assert h[-1].shape == (4, 1, 84, 84)
        return torch.from_numpy(h[-1]).type(torch.float32).to(DEVICE)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(NeuralNetworks):
    def __init__(self, n_states: int, n_actions: int) -> None:
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_states, 400),
            nn.LayerNorm(400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.LayerNorm(300),
            nn.ReLU(),
            nn.Linear(300, n_actions),
            nn.Tanh(),
        ).to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(NeuralNetworks):
    def __init__(self, n_states: int, n_actions: int) -> None:
        super(Critic, self).__init__()
        self.net1 = nn.Sequential(
            nn.Linear(n_states, 400),
            nn.LayerNorm(400),
            nn.ReLU(),
        ).to(DEVICE)

        self.net2 = nn.Sequential(
            nn.Linear(400 + n_actions, 300),
            nn.LayerNorm(300),
            nn.ReLU(),
            nn.Linear(300, 1),
        ).to(DEVICE)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
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
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
        return self


class DDPG(AlgorithmInterface[State, Action]):
    def __init__(self, n_states: int, n_actions: int) -> None:
        self.name = "ddpg"
        self.n_actions = n_actions
        self.n_states = n_states

        self.gamma = 0.99
        self.tau = 1e-2


        self.actor = Actor(self.n_states, self.n_actions)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.actor_loss = nn.MSELoss()

        self.actor_target = (
            Actor(self.n_states, self.n_actions).hard_update_to(self.actor).no_grad()
        )

        self.critic = Critic(self.n_states, self.n_actions)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=1e-3, weight_decay=1e-2
        )
        self.critic_loss = nn.MSELoss()

        self.critic_target = (
            Critic(self.n_states, self.n_actions).hard_update_to(self.critic).no_grad()
        )

        self.replay_buffer_size = int(1e6)
        self.replay_buffer: Deque[Transition] = deque(maxlen=self.replay_buffer_size)

        self.noise_generator = OrnsteinUhlenbeckActionNoise(
            np.zeros(n_actions), sigma=0.2 * np.ones(n_actions)
        ).reset()

        self.mini_batch_size = 64

        self.start_train_ratio = 0.1

        self.times = 0
        self.eval = False

    def on_toggle_eval(self, isEval: bool):
        self.eval = isEval

    def get_mini_batch(self) -> List[Transition]:
        idx = np.random.choice(len(self.replay_buffer), self.mini_batch_size)

        l = list(self.replay_buffer)

        r: List[Transition] = []
        for i in idx:
            r.append(self.replay_buffer[i])

        return r

    def resolve_minibatch_detail(
        self, mini_batch: List[Transition]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states = torch.stack([s for (s, _, _, _, _) in mini_batch])
        assert states.shape == (self.mini_batch_size, self.n_states)

        actions = torch.stack(
            [
                torch.from_numpy(a).type(torch.float32)
                for (_, (a, _), _, _, _) in mini_batch
            ]
        )
        assert actions.shape == (self.mini_batch_size, self.n_actions)

        rewards = torch.stack(
            [torch.tensor(r, dtype=torch.float32) for (_, _, r, _, _) in mini_batch]
        ).unsqueeze(1)
        assert rewards.shape == (self.mini_batch_size, 1)

        next_states = torch.stack([sn for (_, _, _, sn, _) in mini_batch])
        assert next_states.shape == (self.mini_batch_size, self.n_states)

        done = torch.as_tensor(
            [1 if an is None else 0 for (_, _, _, _, an) in mini_batch],
            dtype=torch.int8,
        ).unsqueeze(1)
        assert done.shape == (self.mini_batch_size, 1)

        return (
            states.to(DEVICE),
            actions.to(DEVICE),
            rewards.to(DEVICE),
            next_states.to(DEVICE),
            done.to(DEVICE),
        )

    def set_reporter(self, reporter: Callable[[Dict[str, Any]], None]):
        self.reporter = reporter

    def train(self):
        (states, actions, rewards, next_states, done) = self.resolve_minibatch_detail(
            self.get_mini_batch()
        )

        # next_action_target = self.actor_target(next_states)
        target_q_value = rewards + self.gamma * (1 - done) * self.critic_target(
            next_states, self.actor_target(next_states)
        )
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

        self.reporter(dict(policy_loss=policy_loss, value_loss=value_loss))

    def reset(self):
        self.times = 0
        self.replay_buffer = deque(maxlen=int(1e6))
        self.noise_generator.reset()

    def on_agent_reset(self):
        pass

    @torch.no_grad()
    def take_action(self, state: State) -> Action:
        self.actor.eval()
        act = self.actor(state)
        self.actor.train()
        if not self.eval:
            noise = self.noise_generator.noise()
            act += torch.from_numpy(noise).to(DEVICE)
        return act.cpu().squeeze(0).numpy()

    def on_episode_termination(
        self, sar: Tuple[List[State], List[ActionInfo[Action]], List[Reward]]
    ):
        self.noise_generator.reset()

    def after_step(
        self,
        sar: Tuple[State, ActionInfo[Action], Reward],
        sa: Tuple[State, Optional[ActionInfo[Action]]],
    ):
        (s, a, r) = sar
        (sn, an) = sa
        self.replay_buffer.append((s, a, r, sn, an))

        if len(self.replay_buffer) >= math.ceil(
            self.start_train_ratio * self.replay_buffer_size
        ):
            self.train()

        self.times += 1
