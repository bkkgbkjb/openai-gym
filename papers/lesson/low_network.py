import setup
from utils.episode import Episodes
from utils.replay_buffer import ReplayBuffer
from utils.common import Action
from torch import nn
import torch
from utils.algorithm import Algorithm
from utils.nets import NeuralNetworks, layer_init
from typing import cast
import numpy as np

MAX_TIMESTEPS = 500
ACTION_SCALE = 16.0

Observation = torch.Tensor

State = Observation
Goal = torch.Tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LowCritic(NeuralNetworks):

    def __init__(self, state_dim: int, goal_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim + goal_dim + action_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1)),
        ).to(DEVICE)

        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim

    def forward(self, s: State, g: Goal, a: torch.Tensor) -> torch.Tensor:
        assert s.size(1) == self.state_dim
        assert a.size(1) == self.action_dim
        assert g.size(1) == self.goal_dim
        return self.net(
            torch.cat([s.to(DEVICE), g.to(DEVICE),
                       a.to(DEVICE)], 1))


class LowActor(NeuralNetworks):

    def __init__(self, state_dim: int, goal_dim: int, action_dim: int,
                 action_scale: float):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim + goal_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, action_dim)),
            nn.Tanh(),
        ).to(DEVICE)

        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.action_scale = action_scale

    def forward(self, s: State, g: torch.Tensor) -> torch.Tensor:
        assert s.size(1) == self.state_dim
        assert g.size(1) == self.goal_dim
        assert s.size(0) == g.size(0)
        x = torch.cat([s, g], dim=1)
        act = self.net(x)
        assert act.shape == (s.size(0), self.action_dim)
        return self.action_scale * act


class LowNetwork(Algorithm):

    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
    ):
        self.set_name('low-network')
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.action_scale = ACTION_SCALE

        self.actor = LowActor(self.state_dim, self.goal_dim, self.action_dim,
                              self.action_scale)
        self.actor_target = self.actor.clone().no_grad()

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=2e-4)

        self.critic = LowCritic(self.state_dim, self.goal_dim, self.action_dim)
        self.critic_target = self.critic.clone().no_grad()

        self.critic_loss = nn.MSELoss()

        self.critic_optim = torch.optim.Adam(self.critic.parameters(),
                                             lr=2e-4,
                                             weight_decay=1e-5)

        self.training = True
        self.eps = 0.2
        self.gamma = 0.99

        self.tau = 1e-2

        self.train_times = 0

    def on_toggle_eval(self, isEval: bool):
        self.training = not isEval

    @torch.no_grad()
    def take_action(self, s: torch.Tensor, g: torch.Tensor):
        if self.training and np.random.rand() < self.eps:
            return torch.from_numpy( np.random.uniform(-self.action_scale, self.action_scale,
                                     self.action_dim)).type(torch.float32)

        act = self.actor(s, g).cpu().detach().squeeze()
        if self.training:
            return self.pertub(act)

        return act

    def pertub(self, act: Action):
        act += 0.2 * self.action_scale * torch.randn(self.action_dim)
        return act.clip(-self.action_scale, self.action_scale)

    def sample(self, buffers: ReplayBuffer[Episodes[State]]):
        episodes = buffers.sample(128)

        time_stamps = np.random.randint(MAX_TIMESTEPS - 1, size=128)

        sampled_steps = [
            e.get_step(time_stamps[i]) for i, e in enumerate(episodes)
        ]

        obs = torch.stack([s.state for s in sampled_steps])

        obs_next = torch.stack([s.info['next_obs'] for s in sampled_steps])

        acts = torch.stack([
            cast(Action, s.action).to(DEVICE)
            for s in sampled_steps
        ])

        rgs = torch.stack([s.info['rg'] for s in sampled_steps])

        rg_next = torch.stack([s.info['next_rg'] for s in sampled_steps])

        g = torch.stack([
            s.info['high_act'].to(DEVICE)
            for s in sampled_steps
        ])

        reward = torch.from_numpy(
            -np.linalg.norm(rg_next.cpu().numpy() - g.cpu().numpy(), axis=-1) *
            0.1).type(torch.float32).to(DEVICE)

        g_next = g
        not_done = (torch.norm(
            (rg_next - g_next), dim=1) > 0.1).type(torch.int8).reshape(-1, 1)
        return (obs, obs_next, rgs, rg_next, acts, reward, g, g_next, not_done)

    def train(self, low_buffer: ReplayBuffer[Episodes]):
        (obs, obs_next, rgs, rg_next, acts, reward, g, g_next,
         not_done) = self.sample(low_buffer)

        with torch.no_grad():
            actions_next = self.actor_target(obs_next, g_next)
            q_next_value = self.critic_target(obs_next, g_next,
                                              actions_next).detach()

            target_q_value = (reward +
                              self.gamma * q_next_value * not_done).detach()

        real_q_value = self.critic(obs, g, acts)

        critic_loss = (target_q_value - real_q_value).pow(2).mean()

        actions_real = self.actor(obs, g)
        actor_loss = -self.critic(obs, g, actions_real).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optim.step()

        self.report(
            dict(actor_loss=actor_loss, critic_loss=critic_loss))

        if self.train_times % 3 == 0:
            self.actor_target.soft_update_to(self.actor, self.tau)
            self.critic_target.soft_update_to(self.critic, self.tau)

        self.train_times += 1