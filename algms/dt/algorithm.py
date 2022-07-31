import setup
from utils.algorithm import ActionInfo, Mode
from utils.common import ActionScale
import random
from utils.episode import Episode
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
from dt import DecisionTransformer

from typing import List, Tuple, Any, Optional, Callable, Dict, cast
import numpy as np

from utils.replay_buffer import ReplayBuffer

O = torch.Tensor
Action = torch.Tensor

S = O
Reward = int

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Preprocess(PreprocessI[O, S]):
    def __init__(self, state_mean: torch.Tensor, state_std: torch.Tensor):
        self.state_mean = state_mean
        self.state_std = state_std

    def on_agent_reset(self):
        pass

    def get_current_state(self, h: List[O]) -> S:
        assert len(h) > 0

        return (
            (torch.from_numpy(h[-1]).type(torch.float32) - self.state_mean)
            / self.state_std
        ).to(DEVICE)


class DT(Algorithm[S]):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_scale: ActionScale = 1.0,
    ):
        self.set_name("decision-transformer")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_scale = action_scale

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

        return torch.rand(self.action_dim).uniform_(-1, 1)

    def get_data(self, episodes: List[Episode]):

        for e in episodes:
            # for se in Episode.cut(e, self.k, start = np.random.choice(self.k)):
            #     se.add_info(lambda steps: dict(total_value=np.sum([s.reward if s.reward is not None else 0 for s in steps])))
            e.add_info(
                lambda steps: dict(
                    total_value=np.sum(
                        [s.reward if s.reward is not None else 0 for s in steps]
                    )
                )
            )
            e.compute_returns(1)
            self.total_values.append(e.get_info("total_value"))
            self.replay_buffer.append(e)
        # self.total_values = self.total_values / np.sum(self.total_values)
        self.p_len = [episodes[i].len for i in np.argsort(self.total_values)]
        self.p = self.p_len / np.sum(self.p_len)

    def manual_train(self, info: Dict[str, Any]):
        assert "episodes" in info
        episodes = info["episodes"]

        if self.episodes != episodes:
            self.get_data(episodes)
            self.episodes = episodes

        episodes = self.replay_buffer.sample(self.batch_size, self.p)
        # sub_episodes: List[Episode[S]] = []
        # for e in episodes:
        #     all_se = Episode.cut(e, self.k, start=np.random.choice(self.k))
        #     randint = np.random.choice(len(all_se))
        #     sub_episodes.append(all_se[randint])
        sub_episodes = [
            random.choice(Episode.cut(e, self.k, start=np.random.choice(self.k)))
            for e in episodes
        ]

        states = torch.stack(
            [torch.stack([s.state for s in se.steps]).to(DEVICE) for se in sub_episodes]
        ).to(DEVICE)
        actions = torch.stack(
            [torch.stack([s.action for s in se.steps]).to(DEVICE) for se in sub_episodes]
        ).to(DEVICE)

        # actions = torch.stack([s.action for se in sub_episodes for s in se.steps]).to(
        #     DEVICE
        # )
        rtg = torch.stack(
            [torch.tensor([s.info["return"] for s in se.steps]).to(DEVICE) for se in sub_episodes]
        ).to(DEVICE).unsqueeze(2)
        # rtg = torch.tensor(
        #     [s.info["return"] for se in sub_episodes for s in se.steps]
        # ).to(DEVICE)

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
