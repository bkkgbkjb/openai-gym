import setup
from utils.algorithm import ActionInfo, Mode
from utils.common import ActionScale
from utils.step import NotNoneStep, Step
from utils.transition import Transition, resolve_transitions
from torch import nn
from collections import deque
import torch
from utils.preprocess import PreprocessI
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


class Preprocess(PreprocessI[O, S]):
    def __init__(self):
        pass

    def on_agent_reset(self):
        pass

    def get_current_state(self, h: List[O]) -> S:
        assert len(h) > 0

        # assert h[-1].shape == (4, 1, 84, 84)
        return torch.from_numpy(h[-1]).type(torch.float32).to(DEVICE)


class Actor(NeuralNetworks):
    def __init__(self, state_dim: int, action_dim: int, action_scale: ActionScale, phi=0.05):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim + action_dim, 400)),
            nn.ReLU(),
            layer_init(nn.Linear(400, 300)),
            nn.ReLU(),
            layer_init(nn.Linear(300, action_dim)),
            nn.Tanh(),
        )
        self.phi = phi
        self.action_scale = action_scale

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        s = s.to(DEVICE)
        a = a.to(DEVICE)
        pertub = self.action_scale * self.phi * self.net(torch.cat([s, a], 1))
        return (pertub + a).clamp(-self.action_scale, self.action_scale)


class Critic(NeuralNetworks):
    def __init__(self, state_dim: int, action_dim: int):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim + action_dim, 400)),
            nn.ReLU(),
            layer_init(nn.Linear(400, 300)),
            nn.ReLU(),
            layer_init(nn.Linear(300, 1)),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        s = s.to(DEVICE)
        a = a.to(DEVICE)
        val = self.net(torch.cat([s, a], dim=1))
        return val


class VAE(NeuralNetworks):
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int, action_scale: ActionScale):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(state_dim + action_dim, 750)),
            nn.ReLU(),
            layer_init(nn.Linear(750, 750)),
            nn.ReLU(),
        )

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.decoder = nn.Sequential(
            layer_init(nn.Linear(state_dim + latent_dim, 750)),
            nn.ReLU(),
            layer_init(nn.Linear(750, 750)),
            nn.ReLU(),
            layer_init(nn.Linear(750, action_dim)),
            nn.Tanh(),
        )

        self.latent_dim = latent_dim
        self.action_scale = action_scale

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        s = s.to(DEVICE)
        a = a.to(DEVICE)
        l = self.encoder(torch.cat([s, a], dim=1))

        mean = self.mean(l)
        std = self.log_std(l).clamp(-4, 15).exp()

        z = mean + std * torch.randn_like(std)

        u = self.decode(s, z)
        return u, mean, std

    def decode(self, s: torch.Tensor, z=None):
        s = s.to(DEVICE)
        if z is None:
            z = torch.randn((s.shape[0], self.latent_dim)).clamp(-0.5, 0.5).to(DEVICE)

        a = self.action_scale * self.decoder(torch.cat([s, z], 1))
        return a


class BCQ(Algorithm[S]):
    def __init__(self, state_dim: int, action_dim: int, action_scale: ActionScale = 1.0, discount: float = 0.99, batch_size: int = 128):
        self.set_name("bcq")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_scale = action_scale

        self.discount = discount
        self.tau = 5e-3

        self.lmbda = 0.75
        self.phi = 5e-2
        self.batch_size = batch_size

        self.latent_dim = self.action_dim * 2
        self.actor = Actor(self.state_dim, self.action_dim, self.action_scale, self.phi).to(DEVICE)
        self.actor_target = self.actor.clone().no_grad()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.q1 = Critic(self.state_dim, self.action_dim).to(DEVICE)
        self.q1_target = self.q1.clone().no_grad()

        self.q2 = Critic(self.state_dim, self.action_dim).to(DEVICE)
        self.q2_target = self.q2.clone().no_grad()

        self.q1_loss = nn.MSELoss()
        self.q2_loss = nn.MSELoss()

        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=3e-4)

        self.vae = VAE(self.state_dim, self.action_dim, self.latent_dim, self.action_scale).to(DEVICE)
        self.recon_loss = nn.MSELoss()

        self.vae_optimizer = torch.optim.Adam(self.vae.parameters())
        self.reset()

    def reset(self):
        self.times = 0
        self.transitions = None
        self.replay_buffer = ReplayBuffer(None)

    @torch.no_grad()
    def take_action(self, mode: Mode, state: S) -> Union[ActionInfo, Action]:
        s = state.unsqueeze(0).repeat_interleave(100, 0).to(DEVICE)
        # state = torch.FloatTensor(state.reshape(1,-1).cpu()).repeat(100,1).to(DEVICE)

        a = self.actor(s, self.vae.decode(s))
        q1 = self.q1(s, a)
        act = a[q1.argmax(0)].squeeze(0)
        return act

    def get_data(self, transitions: List[Transition]):

        for transition in transitions:
            self.replay_buffer.append(transition)

    def manual_train(self, info: Dict[str, Any]):
        assert "transitions" in info
        transitions = info["transitions"]

        if self.transitions != transitions:
            self.get_data(transitions)
            self.transitions = transitions

        (states, actions, rewards, next_states, done, _) = resolve_transitions(
            self.replay_buffer.sample(self.batch_size), (self.state_dim,), (self.action_dim,)
        )

        recon, mean, std = self.vae(states, actions)
        recon_loss = self.recon_loss(recon, actions)
        KL_loss = -0.5 * (1 + std.pow(2).log() - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * KL_loss

        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

        with torch.no_grad():
            next_states = next_states.repeat_interleave(10, 0)

            next_actions_repeats = self.actor_target(
                next_states, self.vae.decode(next_states)
            )

            q1_target = self.q1_target(next_states, next_actions_repeats)
            q2_target = self.q2_target(next_states, next_actions_repeats)

            q_target = self.lmbda * torch.min(q1_target, q2_target) + (
                1.0 - self.lmbda
            ) * torch.max(q1_target, q2_target)

            q_target = q_target.reshape(self.batch_size, -1).max(1)[0].unsqueeze(1)

            assert q_target.shape == (self.batch_size, 1)

            q_target = (rewards + (1.0 - done) * self.discount * q_target).detach()

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

        self.report(dict(q_loss=q_loss, vae_loss=vae_loss, actor_loss=actor_loss))

        return self.batch_size
