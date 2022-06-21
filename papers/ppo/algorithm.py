import setup
from utils.step import Step, NotNoneStep
from utils.episode import Episodes
from utils.algorithm import ActionInfo
from utils.transition import (Transition, TransitionTuple)
from torch import nn
import math
from collections import deque
import torch
from utils.preprocess import PreprocessI
from utils.algorithm import Algorithm
from utils.nets import NeuralNetworks, layer_init
from torch.distributions import Categorical
from tqdm.autonotebook import tqdm

from torchvision import transforms as T
from utils.agent import Agent
from gym.spaces import Box
from typing import List, Tuple, Literal, Any, Optional, cast, Callable, Union, Iterable, Dict
import gym
import numpy.typing as npt
from utils.env_sb3 import LazyFrames, PreprocessObservation, FrameStack, resolve_lazy_frames
import numpy as np

Observation = LazyFrames
Action = int

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

        assert h[-1].shape == (4, 84, 84)
        return h[-1]


class ActorCritic(NeuralNetworks):

    def __init__(self, n_actions: int):
        super().__init__()

        self.n_actions = n_actions

        self.base = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, (8, 8), 4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, (4, 4), 2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, (3, 3), 1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(7 * 7 * 64, 512)),
            nn.ReLU(),
        ).to(DEVICE)

        self.actor = nn.Sequential(layer_init(nn.Linear(512,
                                                        n_actions))).to(DEVICE)
        self.critic = layer_init(nn.Linear(512, 1), std=1).to(DEVICE)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        base = self.base(s.to(DEVICE))
        log_action_probs = self.actor(base)
        value = self.critic(base)

        assert log_action_probs.shape == (s.size(0), self.n_actions)
        assert value.shape == (s.size(0), 1)

        return (log_action_probs.cpu(), value.cpu())


class PPO(Algorithm[State]):

    def __init__(
        self,
        n_actions: int,
        sigma: float = 0.1,
        c1: float = 0.5,
        c2: float = 0.01,
        gamma: float = 0.99,
    ):
        self.set_name('ppo')
        self.n_actions = n_actions
        self.network = ActorCritic(n_actions).to(DEVICE)
        self.times = 0

        self.epoch = 4

        self.gae_lambda = .95

        self.gamma = gamma
        self.optimzer = torch.optim.Adam(self.network.parameters(),
                                         2.5e-4,
                                         eps=1e-5)

        self.sigma = sigma
        self.c1 = c1
        self.c2 = c2

        self.episode = Episodes[State]()
        self.batch_size = 64

    @torch.no_grad()
    def take_action(self, state: LazyFrames) -> ActionInfo:

        (log_act_probs,
         value) = self.network(resolve_lazy_frames(state).unsqueeze(0))
        dist = Categorical(logits=log_act_probs)
        act = dist.sample()

        return (
            act,
            {
                "log_prob": dist.log_prob(act),
                "value": value.item()
            },
        )

    def append_step(self, trs: Transition):
        self.episode.append_transition(trs)

    def after_step(self, transition: TransitionTuple[State]):

        trs = Transition(transition)
        self.append_step(trs)

        if self.times != 0 and self.episode.len >= 1024:
            self.train()
            self.episode.clear()

        self.times += 1

    def train(self):
        self.episode.compute_advantages()

        for _ in range(self.epoch):
            for _ in range(32):
                batch = self.episode.sample_non_stop(self.batch_size)

                advs = torch.tensor([
                    cast(float, s.info['advantage']) for s in batch
                ]).type(torch.float32)
                rets = torch.tensor([
                    cast(float, s.info['return']) for s in batch
                ]).type(torch.float32)

                advs = (advs - advs.mean()) / (advs.std() + 1e-8)

                L = self.batch_size

                states = torch.stack(
                    [resolve_lazy_frames(s.state) for s in batch])

                assert states.shape == (L, 4, 84, 84)

                old_acts = torch.cat(
                    [s.action for s in batch])

                assert old_acts.shape == (L, )

                (act_probs, new_vals) = self.network(states)
                new_vals = new_vals.squeeze(1)
                dists = Categorical(logits=act_probs)

                entropy: torch.Tensor = dists.entropy()
                assert entropy.shape == (L, )
                assert entropy.requires_grad

                entropy = entropy.mean()

                new_log_prob = dists.log_prob(old_acts)
                assert new_log_prob.shape == (L, )

                old_log_probs = torch.cat(
                    [s.info["log_prob"] for s in batch])

                assert old_log_probs.shape == (L, )
                assert not old_log_probs.requires_grad

                ratios: torch.Tensor = (new_log_prob - old_log_probs).exp()

                assert ratios.requires_grad

                policy_loss = torch.min(
                    ratios * advs,
                    torch.clamp(ratios, 1 - self.sigma, 1 + self.sigma) * advs,
                )

                assert policy_loss.shape == (L, )
                assert policy_loss.requires_grad

                policy_loss = policy_loss.mean()

                v_loss_unclipped = ((new_vals - rets)**2)

                old_values = torch.tensor(
                    [s.info['value'] for s in batch])

                assert old_values.shape == (L, )

                v_clipped = old_values + \
                    torch.clamp(new_vals - old_values, -self.sigma, self.sigma)
                v_loss_clipped = (v_clipped - rets)**2

                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                value_loss = v_loss_max / 2

                assert value_loss.shape == (L, )

                value_loss = value_loss.mean()

                target = -policy_loss - self.c2 * entropy + self.c1 * value_loss

                self.report({
                    'target': target,
                    'value_loss': value_loss,
                    'policy_loss': policy_loss,
                    'entropy': entropy
                })

                self.optimzer.zero_grad()
                target.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 1)
                self.optimzer.step()

