from turtle import forward
import setup
from utils.algorithm import ActionInfo, Mode
from utils.common import ActionScale, Info
import random
from utils.episode import Episode
from utils.step import NotNoneStep, Step
from utils.transition import Transition, TransitionTuple, resolve_transitions
from torch import nn
from collections import deque
import torch
from utils.preprocess import AllInfo, PreprocessI
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

S = torch.Tensor
RS = torch.Tensor
Reward = int

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Preprocess(PreprocessI[O, RS]):
    def get_current_state(self, all_info: AllInfo) -> RS:
        (o, _, _, _, _) = all_info
        return o[-1].to(DEVICE)


class VAE(NeuralNetworks):
    def __init__(self, state_dim: int, action_dim: int, sequence_len: int):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence_len = sequence_len
        self.input_dim = (self.state_dim + self.action_dim) * self.sequence_len
        self.middle_dim = 256
        self.latent_dim = 32

        self.encoder = nn.Sequential(
            layer_init(nn.Linear(self.input_dim, self.middle_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(self.middle_dim, self.middle_dim)),
            nn.ReLU(),
        )

        self.mean = layer_init(nn.Linear(self.middle_dim, self.latent_dim))
        self.log_std = layer_init(nn.Linear(self.middle_dim, self.latent_dim))

        self.decoder = nn.Sequential(
            layer_init(nn.Linear(self.latent_dim + self.state_dim, self.middle_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(self.middle_dim, self.middle_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(self.middle_dim, self.action_dim)),
            nn.Tanh(),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        # s.shape == (batch_size, seq_len, state_dim)
        # a.shape == (batch_size, seq_len, action_dim)
        s = s.to(DEVICE)
        a = a.to(DEVICE)

        assert s.shape[1] == a.shape[1] == self.sequence_len

        assert s.shape[0] == a.shape[0]
        batch_size = s.shape[0]

        inp = torch.cat([s, a], dim=2).reshape(
            batch_size, self.sequence_len * (self.action_dim + self.state_dim)
        )

        l = self.encoder(inp)
        mean = self.mean(l)
        std = self.log_std(l).clamp(-6, 7).exp()

        z = mean + std * torch.randn_like(std)
        assert z.shape == (batch_size, self.hidden_dim)

        ss = torch.cat(
            [s, z.unsqueeze(1).repeat_interleave(self.sequence_len, dim=1)], dim=2
        )

        acts = self.decoder(
            ss.reshape(batch_size * self.sequence_len, self.state_dim + self.latent_dim)
        ).reshape(batch_size, self.sequence_len, self.action_dim)
        assert acts.shape == (batch_size, self.sequence_len, self.action_dim)

        return acts, mean, std

    def decode(self, s: torch.Tensor, z: Optional[torch.Tensor] = None):
        s = s.to(DEVICE)
        if z is None:
            z = torch.randn((s.shape[0], self.latent_dim)).clamp(-0.5, 0.5)
        z = z.to(DEVICE)

        a = self.decoder(torch.cat([s, z], 1))
        return a


class AE(NeuralNetworks):
    def __init__(
        self, state_dim: int, action_dim: int, sequence_len: int, latent_dim: int
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence_len = sequence_len
        self.input_dim = (self.state_dim + self.action_dim) * self.sequence_len
        self.middle_dim = 256
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            layer_init(nn.Linear(self.input_dim, self.middle_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(self.middle_dim, self.middle_dim)),
            nn.ReLU(),
        )

        self.latent = nn.Sequential(
            layer_init(nn.Linear(self.middle_dim, self.latent_dim)), nn.Tanh()
        )

        self.decoder = nn.Sequential(
            layer_init(nn.Linear(self.latent_dim + self.state_dim, self.middle_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(self.middle_dim, self.middle_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(self.middle_dim, self.action_dim)),
            nn.Tanh(),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        # s.shape == (batch_size, seq_len, state_dim)
        # a.shape == (batch_size, seq_len, action_dim)
        s = s.to(DEVICE)
        a = a.to(DEVICE)

        assert s.shape[1] == a.shape[1] == self.sequence_len

        assert s.shape[0] == a.shape[0]
        batch_size = s.shape[0]

        inp = torch.cat([s, a], dim=2).reshape(
            batch_size, self.sequence_len * (self.action_dim + self.state_dim)
        )

        l = self.encoder(inp)
        z = self.latent(l)
        assert z.shape == (batch_size, self.latent_dim)

        ss = torch.cat(
            [s, z.unsqueeze(1).repeat_interleave(self.sequence_len, dim=1)], dim=2
        )

        acts = self.decoder(
            ss.reshape(batch_size * self.sequence_len, self.state_dim + self.latent_dim)
        ).reshape(batch_size, self.sequence_len, self.action_dim)
        assert acts.shape == (batch_size, self.sequence_len, self.action_dim)

        return acts, z

    def decode(self, s: torch.Tensor, z: torch.Tensor):
        s = s.to(DEVICE)
        z = z.to(DEVICE)
        assert torch.logical_and(-1 <= z, z <= 1).all()

        a = self.decoder(torch.cat([s, z], 1))
        return a


class Simple(Algorithm[S]):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
    ):
        self.set_name("simple-ae")
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.batch_size = 128
        self.latent_dim = 32

        self.k = 40
        self.ae = AE(self.state_dim, self.action_dim, self.k, self.latent_dim).to(
            DEVICE
        )

        self.ae_optimizer = torch.optim.Adam(
            self.ae.parameters(), lr=1e-4, weight_decay=1e-4
        )
        self.recon_loss = nn.MSELoss()

        self.z_star = torch.rand(self.latent_dim).uniform_(-1, 1).to(DEVICE)
        self.z_star.requires_grad = True
        self.z_star_optimizer = torch.optim.Adam(
            [self.z_star], lr=1e-4, weight_decay=1e-4
        )

        self.z_star_loss = torch.nn.CrossEntropyLoss()

        self.reset()

    def reset(self):
        self.times = 0
        self.episodes = None
        self.last_z = torch.clone(self.z_star)
        self.replay_buffer = ReplayBuffer[Tuple[Episode[S], Episode[S], bool]](None)

    @torch.no_grad()
    def take_action(self, mode: Mode, state: S) -> Union[ActionInfo, Action]:

        return self.ae.decode(
            state.unsqueeze(0), torch.tanh(self.z_star).unsqueeze(0)
        ).squeeze()

    def get_data(self, episodes: List[Episode]):

        all_ses: List[Episode[S]] = []
        for e in episodes:
            for se in Episode.cut(e, self.k, start=np.random.choice(self.k)):
                se.add_info(
                    lambda steps: dict(total_value=np.sum([s.reward for s in steps]))
                )
                all_ses.append(se)

        for se in all_ses:
            pair = random.choice(all_ses)

            se_value = se.get_info("total_value")
            pair_value = pair.get_info("total_value")

            assert se_value > 0 and pair_value > 0

            max_value = np.max([se_value, pair_value])
            min_value = np.min([se_value, pair_value])

            se_is_max = se_value > pair_value

            if (max_value - min_value) / min_value <= 1 / 3:
                continue

            self.replay_buffer.append((se, pair, True if se_is_max else False))

    def sample(self):
        episode_pairs = self.replay_buffer.sample(self.batch_size)

        episode_1 = [e for (e, _, _) in episode_pairs]
        episode_2 = [e for (_, e, _) in episode_pairs]
        e1_lt_e2 = [l for (_, _, l) in episode_pairs]

        states_1 = torch.stack(
            [torch.stack([s.state for s in e.steps]) for e in episode_1]
        )
        actions_1 = torch.stack(
            [torch.stack([s.action for s in e.steps]) for e in episode_1]
        )
        states_2 = torch.stack(
            [torch.stack([s.state for s in e.steps]) for e in episode_2]
        )
        actions_2 = torch.stack(
            [torch.stack([s.action for s in e.steps]) for e in episode_2]
        )

        return (states_1, actions_1, states_2, actions_2, e1_lt_e2)

    def manual_train(self, info: Dict[str, Any]):
        assert "episodes" in info
        episodes = info["episodes"]

        if self.episodes != episodes:
            self.get_data(episodes)
            self.episodes = episodes

        (states_1, actions_1, states_2, actions_2, e1_lt_e2) = self.sample()
        one_lt_two_target = torch.stack(
            [
                torch.tensor([1.0, 0.0] if l else [0.0, 1.0], dtype=torch.float32)
                for l in e1_lt_e2
            ]
        ).to(DEVICE)

        (pred_actions1, z1) = self.ae(states_1, actions_1)
        (pred_actions2, z2) = self.ae(states_2, actions_2)

        recon_loss_1 = self.recon_loss(
            actions_1.to(DEVICE).reshape(self.batch_size * self.k, self.action_dim),
            pred_actions1.reshape(self.batch_size * self.k, self.action_dim),
        )
        recon_loss_2 = self.recon_loss(
            actions_2.to(DEVICE).reshape(self.batch_size * self.k, self.action_dim),
            pred_actions2.reshape(self.batch_size * self.k, self.action_dim),
        )

        recon_loss = recon_loss_1 + recon_loss_2

        dist1 = torch.norm(
            z1 - self.z_star,
            dim=1,
        )
        dist2 = torch.norm(
            z2 - self.z_star,
            dim=1,
        )
        assert dist1.shape == dist2.shape == (self.batch_size,)

        z_star_loss = self.z_star_loss(
            torch.cat([dist1.unsqueeze(1), dist2.unsqueeze(1)], dim=1),
            one_lt_two_target,
        )

        loss = z_star_loss + recon_loss

        self.ae_optimizer.zero_grad()
        loss.backward()
        self.ae_optimizer.step()

        (_, z1_new) = self.ae(states_1, actions_1)
        (_, z2_new) = self.ae(states_2, actions_2)
        dist1_new = torch.norm(
            z1_new - self.z_star,
            dim=1,
        )
        dist2_new = torch.norm(
            z2_new - self.z_star,
            dim=1,
        )
        assert dist1_new.shape == dist2_new.shape == (self.batch_size,)

        z_star_loss_new = self.z_star_loss(
            torch.cat([dist1_new.unsqueeze(1), dist2_new.unsqueeze(1)], dim=1),
            one_lt_two_target,
        )

        self.z_star_optimizer.zero_grad()
        z_star_loss_new.backward()
        self.z_star_optimizer.step()

        delta_z = torch.abs(self.z_star - self.last_z).mean()
        self.last_z = torch.clone(self.z_star)

        self.report(
            dict(recon_loss=recon_loss, z_loss=z_star_loss_new, delta_z=delta_z)
        )

        return self.batch_size
