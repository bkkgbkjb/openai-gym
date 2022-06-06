import setup
from utils.common import (
    ActionInfo,
    AllowedState,
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
from torch.utils.data import DataLoader

from typing import List, Tuple, Any, Optional, Callable, Dict, cast
import numpy as np

from utils.replay_buffer import ReplayBuffer

O = torch.Tensor
Action = torch.Tensor

S = O
Reward = int

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Preprocess(Preprocess[O]):

    def __init__(self):
        pass

    def on_agent_reset(self):
        pass

    def get_current_state(self, h: List[O]) -> S:
        assert len(h) > 0

        # assert h[-1].shape == (4, 1, 84, 84)
        return torch.from_numpy(h[-1]).type(torch.float32).to(DEVICE)


class Actor(NeuralNetworks):

    def __init__(self, state_dim: int, action_dim: int, phi=0.05):
        super(Actor, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, 400),
                                 nn.ReLU(), nn.Linear(400, 300), nn.ReLU(),
                                 nn.Linear(300, action_dim), nn.Tanh())
        self.phi = phi

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        pertub = self.phi * self.net(torch.cat([s, a], 1))
        return (pertub + a).clamp(-1.0, 1.0)


class Critic(NeuralNetworks):

    def __init__(self, state_dim: int, action_dim: int):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        val = self.net(torch.cat([s, a], dim=1))
        return val


class VAE(NeuralNetworks):

    def __init__(self, state_dim: int, action_dim: int, latent_dim: int):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim + action_dim, 750),
                                     nn.ReLU(), nn.Linear(750, 750), nn.ReLU())

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.decoder = nn.Sequential(nn.Linear(state_dim + latent_dim, 750),
                                     nn.ReLU(), nn.Linear(750, 750), nn.ReLU(),
                                     nn.Linear(750, action_dim), nn.Tanh())

        self.latent_dim = latent_dim

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        l = self.encoder(torch.cat([s, a], dim=1))

        mean = self.mean(l)
        std = self.log_std(l).clamp(-4, 15).exp()

        z = mean + std * torch.randn_like(std)

        u = self.decode(s, z)
        return u, mean, std

    def decode(self, s: torch.Tensor, z=None):
        if z is None:
            z = torch.randn(
                (s.shape[0], self.latent_dim)).clamp(-0.5, 0.5).to(DEVICE)

        a = self.decoder(torch.cat([s, z], 1))
        return a


class BCQ(Algorithm):

    def __init__(self, state_dim: int, action_dim: int):
        self.name = 'bcq'
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.discount = 0.99
        self.tau = 5e-3

        self.lmbda = 0.75
        self.phi = 5e-2

        self.latent_dim = self.action_dim * 2
        self.actor = Actor(self.state_dim, self.action_dim,
                           self.phi).to(DEVICE)
        self.actor_target = self.actor.clone().no_grad()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=1e-3)

        self.q1 = Critic(self.state_dim, self.action_dim).to(DEVICE)
        self.q1_target = self.q1.clone().no_grad()

        self.q2 = Critic(self.state_dim, self.action_dim).to(DEVICE)
        self.q2_target = self.q2.clone().no_grad()

        self.q1_loss = nn.MSELoss()
        self.q2_loss = nn.MSELoss()

        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=1e-3)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=1e-3)

        self.vae = VAE(self.state_dim, self.action_dim,
                       self.latent_dim).to(DEVICE)
        self.recon_loss = nn.MSELoss()

        self.vae_optimizer = torch.optim.Adam(self.vae.parameters())
        self.reset()

    def reset(self):
        self.times = 0
        self.replay_buffer = ReplayBuffer((self.state_dim, ),
                                          (self.action_dim, ), None)

    @torch.no_grad()
    def take_action(self, state: S) -> Union[ActionInfo, Action]:
        s = state.unsqueeze(0).repeat_interleave(100, 0).to(DEVICE)
        # state = torch.FloatTensor(state.reshape(1,-1).cpu()).repeat(100,1).to(DEVICE)

        a = self.actor(s, self.vae.decode(s))
        q1 = self.q1(s, a)
        act = a[q1.argmax(0)].squeeze(0).cpu().numpy()
        return act

    def on_init(self, info: Dict[str, Any]):
        assert 'dataloader' in info
        self.dataloader = info['dataloader']

        for (states, actions, rewards, next_states, dones) in self.dataloader:
            for (s, a, r, sn, done) in zip(states, actions, rewards,
                                           next_states, dones):
                self.replay_buffer.append(
                    (s, (a.numpy(), dict()), r.item(), sn, done.item() == 1))

    def manual_train(self):

        (states, actions, rewards, next_states,
         done) = ReplayBuffer.resolve(self.replay_buffer.sample(100),
                                      (self.state_dim, ), (self.action_dim, ))

        recon, mean, std = self.vae(states, actions)
        recon_loss = self.recon_loss(recon, actions)
        KL_loss = -0.5 * (1 + std.pow(2).log() - mean.pow(2) -
                          std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * KL_loss

        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

        with torch.no_grad():
            next_states = next_states.repeat_interleave(10, 0)

            next_actions_repeats = self.actor_target(
                next_states, self.vae.decode(next_states))

            q1_target = self.q1_target(next_states, next_actions_repeats)
            q2_target = self.q2_target(next_states, next_actions_repeats)

            q_target = self.lmbda * torch.min(q1_target, q2_target) + (
                1.0 - self.lmbda) * torch.max(q1_target, q2_target)

            q_target = q_target.reshape(100, -1).max(1)[0].unsqueeze(1)

            assert q_target.shape == (100, 1)

            q_target = (rewards +
                        (1.0 - done) * self.discount * q_target).detach()

        q1 = self.q1(states, actions)
        q2 = self.q2(states, actions)

        assert not q_target.requires_grad
        q_loss = self.q1_loss(q1, q_target) + self.q2_loss(q2, q_target)

        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        q_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # sampled_actions = self.vae.sample(states)
        sampled_actions = self.actor(states, self.vae.decode(states))
        # pertubed_actions = (sampled_actions + pertubs).clamp(-1, 1)

        actor_loss = -self.q1(states, sampled_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.q1_target.soft_update_to(self.q1, self.tau)
        self.q2_target.soft_update_to(self.q2, self.tau)
        self.actor_target.soft_update_to(self.actor, self.tau)

        self.report(
            dict(q_loss=q_loss, vae_loss=vae_loss, actor_loss=actor_loss))

        return 100
