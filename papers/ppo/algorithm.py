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
from utils.preprocess import PreprocessInterface
from utils.algorithm import AlgorithmInterface
import plotly.graph_objects as go
from torch.distributions import Categorical
from tqdm.autonotebook import tqdm

from torchvision import transforms as T
from utils.agent import Agent
from gym.spaces import Box
from typing import List, Tuple, Literal, Any, Optional, cast, Callable, Union, Iterable, Dict
import gym
import numpy.typing as npt
from utils.env import PreprocessObservation, FrameStack, resolve_lazy_frames
import numpy as np

Observation = torch.Tensor
Action = int

State = torch.Tensor
Reward = int

Transition = TransitionGeneric[State, Action]
Step = StepGeneric[State, ActionInfo[Action]]
NotNoneStep = NotNoneStepGeneric[State, ActionInfo[Action]]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ActorCritic(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()

        self.n_actions = n_actions

        self.base = nn.Sequential(
            nn.Conv2d(4, 32, (8, 8), 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            # nn.Linear(256, n_actions),
            # nn.Softmax(dim=1),
        ).to(DEVICE)

        self.actor = nn.Sequential(nn.Linear(512, n_actions), nn.Softmax(dim=1)).to(
            DEVICE
        )
        self.critic = nn.Linear(512, 1).to(DEVICE)

    def forward(self, s: State) -> Tuple[torch.Tensor, torch.Tensor]:
        base = self.base(s.to(DEVICE))
        action_probs = self.actor(base)
        value = self.critic(base)

        assert action_probs.shape == (s.size(0), self.n_actions)
        assert value.shape == (s.size(0), 1)

        return (action_probs.cpu(), value.cpu())


class PPO(AlgorithmInterface[State, Action]):
    def __init__(
        self,
        n_actions: int,
        sigma: float = 0.1,
        c1: float = 0.5,
        c2: float = 0.01,
        gamma: float = 0.99,
    ):
        self.frame_skip = 0
        self.name = "ppo"
        self.n_actions = n_actions
        # self.actor = Actor(n_actions).to(DEVICE)
        # self.critic = Critic().to(DEVICE)
        self.network = ActorCritic(n_actions).to(DEVICE)
        self.times = -1

        self.epoch = 4

        self.gamma = gamma
        self.update_freq = 250
        self.optimzer = torch.optim.Adam(
            self.network.parameters(), 1e-4, eps=1e-5)

        self.sigma = sigma
        self.c1 = c1
        self.c2 = c2

        self.memory: List[Step] = []
        self.batch_size = 32
        self.reporter = None

    def set_reporter(self, reporter: Callable[[Dict[Any, Any]], None]):
        self.reporter = reporter

    def on_reset(self):
        pass

    def allowed_actions(self, state: State) -> List[Action]:
        return list(range(self.n_actions))

    @torch.no_grad()
    def take_action(self, state: State) -> ActionInfo[Action]:

        (act_probs, value) = self.network(resolve_lazy_frames(state))
        dist = Categorical(act_probs)
        act = dist.sample()

        return (
            cast(int, act),
            {"log_prob": dist.log_prob(act), "value": value},
        )

    def append_step(
        self,
        s: State,
        a: ActionInfo[Action],
        r: Reward,
        sn: State,
        an: Optional[ActionInfo[Action]],
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

    def after_step(
        self,
        sar: Tuple[State, ActionInfo[Action], Reward],
        sa: Tuple[State, Optional[ActionInfo[Action]]],
    ):
        self.times += 1
        (s, a, r) = sar
        (sn, an) = sa
        self.append_step(s, a, r, sn, an)

        if self.times != 0 and len(self.memory) == 1024:
            self.train()
            self.memory: List[Step] = []

    @property
    def no_stop_step(self) -> Iterable[NotNoneStep]:

        return (
            (s, a, cast(Reward, r)) for (s, a, r) in self.memory[:-1] if a is not None
        )

    @torch.no_grad()
    def compute_advantages_and_returns(self) -> Tuple[torch.Tensor, torch.Tensor]:

        values = [i["value"].item() for (_, (_, i), _) in self.no_stop_step]

        returns = [float("nan") for _ in range(len(values))]

        (_, la, _) = self.memory[-1]

        next_is_stop = la is None
        next_value = 0 if next_is_stop else la[1]["value"].item()

        i = 0
        for (_, a, _) in reversed(self.memory[:-1]):
            if a is None:
                assert not next_is_stop
                next_is_stop = True
                next_value = 0
            else:
                (_, info) = a
                returns[-(i + 1)] = (
                    info["value"].item() + 0
                    if next_is_stop
                    else self.gamma * next_value
                )
                next_is_stop = False
                next_value = returns[-(i + 1)]
                i += 1

        values = torch.tensor(values, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        return returns - values, returns

    def train(self):
        (advantages, returns) = self.compute_advantages_and_returns()
        memory = list(self.no_stop_step)

        for _ in range(self.epoch):
            for _ in range(math.ceil(1024 / self.batch_size)):
                batch_index = np.random.choice(len(memory), self.batch_size)

                batch: List[NotNoneStep] = []
                for i in batch_index:
                    batch.append(memory[i])

                advs = advantages[batch_index]
                rets = returns[batch_index]

                L = self.batch_size

                states = torch.cat([resolve_lazy_frames(s)
                                   for (s, _, _) in batch])

                states.shape == (L, 4, 84, 84)

                old_acts = torch.tensor([a for (_, (a, _), _) in batch])

                assert old_acts.shape == (L,)

                (act_probs, new_vals) = self.network(states)
                dists = Categorical(act_probs)

                entropy: torch.Tensor = dists.entropy()
                assert entropy.shape == (L,)
                assert entropy.requires_grad

                entropy = entropy.mean()

                new_log_prob = dists.log_prob(old_acts)
                assert new_log_prob.shape == (L,)

                old_log_probs = torch.cat([i["log_prob"]
                                          for (_, (_, i), _) in batch])

                assert old_log_probs.shape == (L,)
                assert not old_log_probs.requires_grad

                ratios: torch.Tensor = (new_log_prob - old_log_probs).exp()

                assert ratios.requires_grad

                policy_loss = torch.min(
                    ratios * advs,
                    torch.clamp(ratios, 1 - self.sigma, 1 + self.sigma) * advs,
                )

                assert policy_loss.shape == (L,)
                assert policy_loss.requires_grad

                policy_loss = policy_loss.mean()

                value_loss = ((new_vals.squeeze(1) - rets) ** 2) / 2

                assert value_loss.shape == (L,)

                value_loss = value_loss.mean()

                target = -policy_loss - self.c2 * entropy + self.c1 * value_loss
                # assert target.shape == (L,)

                self.optimzer.zero_grad()
                target.mean().backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 1)
                self.optimzer.step()

                self.reporter and self.reporter({
                    'target': target,
                    'value_loss': value_loss,
                    'policy_loss': policy_loss,
                    'entropy': entropy
                })

    def on_termination(
        self, sar: Tuple[List[State], List[ActionInfo[Action]], List[Reward]]
    ):
        (s, a, r) = sar
        assert len(s) == len(a) + 1
        assert len(s) == len(r) + 1
        pass


class RandomAlgorithm(AlgorithmInterface[State, Action]):
    def __init__(self, n_actions: int):
        self.name = "random"
        self.frame_skip = 0
        self.n_actions = n_actions
        self.times = -1
        self.action_keeps = 10
        self.reset()

    def reset(self):
        self.last_action = None

    def on_reset(self):
        self.reset()

    def allowed_actions(self, state: State) -> List[Action]:
        return list(range(self.n_actions))

    def take_action(self, state: State) -> Action:
        self.times += 1

        if self.times % self.action_keeps == 0:
            act = np.random.choice(self.allowed_actions(state))
            self.last_action = act
            return act

        assert self.last_action is not None
        return self.last_action

    def after_step(
        self,
        sar: Tuple[State, Action, Reward],
        sa: Tuple[State, Optional[Action]],
    ):
        pass

    def on_termination(self, sar: Tuple[List[State], List[Action], List[Reward]]):
        (s, a, r) = sar
        assert len(s) == len(a) + 1
        assert len(s) == len(r) + 1
        pass


class Preprocess(PreprocessInterface[Observation, Action, State]):
    def __init__(self):
        pass

    def on_reset(self):
        pass

    def get_current_state(self, h: List[Observation]) -> State:
        assert len(h) > 0

        assert h[-1].shape == (4, 1, 84, 84)
        return h[-1]
