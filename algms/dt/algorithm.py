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
from dt import DecisionTransformer

from typing import List, Tuple, Any, Optional, Callable, Dict, cast
import numpy as np

from utils.replay_buffer import ReplayBuffer

O = torch.Tensor
Action = torch.Tensor

S = torch.Tensor
RS = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
Reward = int

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Preprocess(PreprocessI[O, RS]):
    def __init__(
        self,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        action_dim: int,
        rtg: float,
    ):
        self.state_mean = state_mean
        self.state_std = state_std
        self.action_dim = action_dim
        self.k = 20
        self.rtg = rtg

    def get_current_state(self, all_info: AllInfo) -> RS:
        (o, s, a, r, i) = all_info
        assert len(o) == len(s) + 1
        assert len(s) == len(a) == len(r) == len(i)

        s = self.pad(
            (torch.stack(o) - self.state_mean) / self.state_std,
            torch.stack(a + [torch.zeros(self.action_dim)]),
            torch.tensor(self.rtg).float() - torch.as_tensor([0.0] + r).float(),
        )
        return s

    def pad(
        self, states: torch.Tensor, actions: torch.Tensor, rtgs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        L = states.shape[0]

        state_dim = states.shape[1]
        action_dim = actions.shape[1]

        pad_len = self.k - L
        if pad_len <= 0:
            masks = torch.ones(self.k).type(torch.int64)
            timesteps = torch.arange(L - self.k, L).type(torch.int64)
            return (
                states[-self.k :].to(DEVICE),
                actions[-self.k :].to(DEVICE),
                masks.to(DEVICE),
                timesteps.to(DEVICE),
                rtgs[-self.k :].unsqueeze(1).to(DEVICE),
            )

        new_states = torch.cat(
            [
                torch.zeros(
                    (
                        pad_len,
                        state_dim,
                    )
                ),
                states,
            ]
        )
        new_actions = torch.cat(
            [
                torch.zeros(
                    (
                        pad_len,
                        action_dim,
                    )
                ),
                actions,
            ]
        )

        masks = torch.cat([torch.zeros(pad_len), torch.ones(L)]).type(torch.int64)
        timesteps = torch.cat([torch.zeros(pad_len), torch.arange(0, L)]).type(
            torch.int64
        )
        rtgs = torch.cat([torch.zeros(pad_len), rtgs])
        return (
            new_states.to(DEVICE),
            new_actions.to(DEVICE),
            masks.to(DEVICE),
            timesteps.to(DEVICE),
            rtgs.unsqueeze(1).to(DEVICE),
        )


class DT(Algorithm[S]):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
    ):
        self.set_name("decision-transformer")
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.batch_size = 64

        self.latent_dim = 128
        self.loss = nn.MSELoss()
        self.k = 20

        self.dt = DecisionTransformer(
            self.state_dim,
            self.action_dim,
            self.latent_dim,
            self.k,
            1000,
            n_layer=3,
            n_head=1,
            n_inner=4 * self.latent_dim,
            n_positions=1024,
            resid_pdrop=0.1,
            activation_function="relu",
            attn_pdrop=0.1,
        ).to(DEVICE)

        self.optimizer = torch.optim.AdamW(
            self.dt.parameters(), lr=1e-4, weight_decay=1e-4
        )
        self.reset()

    def reset(self):
        self.times = 0
        self.episodes = None
        self.total_values: List[float] = []
        self.replay_buffer = ReplayBuffer[Episode[S]](None)

    @torch.no_grad()
    def take_action(self, mode: Mode, state: S) -> Union[ActionInfo, Action]:

        states, actions, masks, timesteps, rtgs = state

        with torch.no_grad():
            _, actions, _ = self.dt.forward(
                states.unsqueeze(0),
                actions.unsqueeze(0),
                rtgs.unsqueeze(0),
                timesteps.unsqueeze(0),
                masks.unsqueeze(0),
            )
        return actions[0, -1]

    def after_step(self, mode: Mode, transition: TransitionTuple[S]):
        (s1, _) = transition
        self.times += 1

    def on_episode_termination(
        self, mode: Mode, sari: Tuple[List[S], List[Action], List[Reward], List[Info]]
    ):
        self.times = 0

    def get_data(self, episodes: List[Episode]):

        for e in episodes:
            e.add_info(
                lambda steps: dict(
                    total_value=np.sum([s.reward for s in steps])
                )
            )
            e.compute_returns(1)
            self.total_values.append(e.get_info("total_value"))
            self.replay_buffer.append(e)
        self.p_len = [episodes[i].len for i in np.argsort(self.total_values)]
        self.p = self.p_len / np.sum(self.p_len)

    def manual_train(self, info: Dict[str, Any]):
        assert "episodes" in info
        episodes = info["episodes"]

        if self.episodes != episodes:
            self.get_data(episodes)
            self.episodes = episodes

        episodes = self.replay_buffer.sample(self.batch_size, self.p)
        sub_episodes = [
            random.choice(Episode.cut(e, self.k, start=np.random.choice(self.k)))
            for e in episodes
        ]

        states = torch.stack(
            [torch.stack([s.state for s in se.steps]).to(DEVICE) for se in sub_episodes]
        ).to(DEVICE)
        actions = torch.stack(
            [
                torch.stack([s.action for s in se.steps]).to(DEVICE)
                for se in sub_episodes
            ]
        ).to(DEVICE)

        rtg = (
            torch.stack(
                [
                    torch.tensor([s.info["return"] for s in se.steps]).to(DEVICE)
                    for se in sub_episodes
                ]
            )
            .to(DEVICE)
            .unsqueeze(2)
        ) / 1000

        timesteps = torch.stack(
            [
                torch.arange(se.get_info("start"), se.get_info("end")).to(DEVICE)
                for se in sub_episodes
            ]
        ).to(DEVICE)

        _, action_preds, _ = self.dt.forward(states, actions, rtg, timesteps)

        loss = self.loss(
            action_preds.reshape(-1, self.action_dim),
            actions.detach().reshape(-1, self.action_dim),
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.report(dict(loss=loss))

        return self.batch_size
