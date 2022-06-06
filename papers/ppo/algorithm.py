import setup
from utils.common import (
    ActionInfo,
    Step,
    NotNoneStep,
    Transition,
    TransitionTuple,
)
from torch import nn
import math
from collections import deque
import torch
from utils.preprocess import Preprocess
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

Observation = torch.Tensor
Action = int

State = torch.Tensor
Reward = float

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

    def forward(self, s: State) -> Tuple[torch.Tensor, torch.Tensor]:
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
        self.name = "ppo"
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

        self.memory: List[Step] = []
        self.batch_size = 32

    @torch.no_grad()
    def take_action(self, state: LazyFrames) -> ActionInfo:

        (log_act_probs,
         value) = self.network(resolve_lazy_frames(state).unsqueeze(0))
        dist = Categorical(logits=log_act_probs)
        act = dist.sample()

        return (
            act.numpy(),
            {
                "log_prob": dist.log_prob(act),
                "value": value.item()
            },
        )

    def append_step(
        self,
        s: LazyFrames,
        a: ActionInfo,
        r: Reward,
        sn: LazyFrames,
        an: Optional[ActionInfo],
    ):
        if len(self.memory) == 0:
            self.memory.extend([(s, a, r), (sn, an, None)])
            return

        (ls, la, lr) = self.memory[-1]
        if la is not None:
            assert lr is None
            self.memory[-1] = (ls, la, r)
            self.memory.append((sn, an, None))
            return

        self.memory.extend([(s, a, r), (sn, an, None)])

    def after_step(self, transition: TransitionTuple[State]):
        (s, a, r, sn, an) = transition

        assert isinstance(an, tuple) or an is None
        assert isinstance(s, LazyFrames)
        assert isinstance(sn, LazyFrames)

        self.append_step(s, a, r, sn, an)

        if self.times != 0 and len(self.memory) >= 1024:
            self.train()
            self.memory: List[Step] = []

        self.times += 1

    @property
    def non_stop_step(self) -> Iterable[NotNoneStep]:

        return ((s, a, cast(Reward, r)) for (s, a, r) in self.memory[:-1]
                if a is not None)

    @torch.no_grad()
    def compute_advantages_and_returns_gae(
            self) -> Tuple[torch.Tensor, torch.Tensor]:

        values = [i["value"] for (_, (_, i), _) in self.non_stop_step]

        advs = [float("nan") for _ in range(len(values))]

        (_, la, _) = self.memory[-1]

        next_is_stop = la is None
        next_value = 0 if next_is_stop else la[1]["value"]
        lastgaelambda = 0
        i = 0
        for j in reversed(range(len(self.memory[:-1]))):
            (_, a, r) = self.memory[j]
            if a is None:
                assert not next_is_stop
                next_is_stop = True
                next_value = 0
            else:
                assert r is not None

                delta = r + (0 if next_is_stop else self.gamma *
                             next_value) - values[-(i + 1)]
                advs[-(i+1)] = delta + \
                    (0 if next_is_stop else self.gamma *
                     self.gae_lambda * lastgaelambda)
                next_is_stop = False
                lastgaelambda = advs[-(i + 1)]
                next_value = values[-(i + 1)]
                i += 1

        values = torch.tensor(values, dtype=torch.float32)
        advs = torch.tensor(advs, dtype=torch.float32)

        return advs, advs + values

    @torch.no_grad()
    def compute_advantages_and_returns(
            self) -> Tuple[torch.Tensor, torch.Tensor]:

        values = [i["value"] for (_, (_, i), _) in self.non_stop_step]

        returns = [float("nan") for _ in range(len(values))]

        (_, la, _) = self.memory[-1]

        next_is_stop = la is None
        next_value = 0 if next_is_stop else la[1]["value"]

        i = 0
        for j in reversed(range(len(self.memory[:-1]))):
            # for (_, a, r) in reversed(self.memory[:-1]):
            (_, a, r) = self.memory[j]
            if a is None:
                assert not next_is_stop
                next_is_stop = True
                next_value = 0
            else:
                assert r is not None
                returns[-(i + 1)] = (
                    r + (0 if next_is_stop else self.gamma * next_value))
                next_is_stop = False
                next_value = returns[-(i + 1)]
                i += 1

        values = torch.tensor(values, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        return returns - values, returns

    def train(self):
        (advantages, returns) = self.compute_advantages_and_returns_gae()
        memory = list(self.non_stop_step)

        for _ in range(self.epoch):
            for _ in range(math.ceil(len(self.memory) / self.batch_size)):
                batch_index = np.random.choice(len(memory), self.batch_size)

                batch: List[NotNoneStep] = []
                for i in batch_index:
                    batch.append(memory[i])

                advs = advantages[batch_index]
                advs = (advs - advs.mean()) / (advs.std() + 1e-8)
                rets = returns[batch_index]

                L = self.batch_size

                states = torch.stack(
                    [resolve_lazy_frames(s) for (s, _, _) in batch])

                assert states.shape == (L, 4, 84, 84)

                old_acts = torch.tensor([a for (_, (a, _), _) in batch
                                         ]).squeeze(1)

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
                    [i["log_prob"] for (_, (_, i), _) in batch])

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
                    [i['value'] for (_, (_, i), _) in batch])

                assert old_values.shape == (L, )

                v_clipped = old_values + \
                    torch.clamp(new_vals - old_values, -self.sigma, self.sigma)
                v_loss_clipped = (v_clipped - rets)**2

                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                value_loss = v_loss_max / 2

                assert value_loss.shape == (L, )

                value_loss = value_loss.mean()

                target = -policy_loss - self.c2 * entropy + self.c1 * value_loss
                # assert target.shape == (L,)

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


class Preprocess(Preprocess[Observation, State]):

    def __init__(self):
        pass

    def on_agent_reset(self):
        pass

    def get_current_state(self, h: List[Observation]) -> State:
        assert len(h) > 0

        assert h[-1].shape == (4, 84, 84)
        return h[-1]
