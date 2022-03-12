from importlib_metadata import requires
import setup
from utils.common import ActionInfo, Step, Episode, TransitionGeneric
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
from typing import List, Tuple, Literal, Any, Optional, cast, Callable, Union, Iterable
import gym
import numpy.typing as npt
from utils.env import PreprocessObservation, FrameStack, resolve_lazy_frames
import numpy as np

Observation = torch.Tensor
Action = int

State = torch.Tensor
Reward = int

Transition = TransitionGeneric[State, Action]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Actor(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()

        self.n_actions = n_actions

        self.network = nn.Sequential(
            nn.Conv2d(4, 32, (8, 8), 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), 2),
            nn.ReLU(),
            nn.Conv2d(64, 32, (3, 3), 1),
            nn.Flatten(),
            nn.Linear(7 * 7 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
            nn.Softmax(dim=1),
        )

    def forward(self, s: State) -> torch.Tensor:
        rlt = cast(torch.Tensor, self.network(
            s.to(DEVICE)))
        assert rlt.shape == (s.size(0), self.n_actions)
        return rlt.cpu()


class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(4, 32, (8, 8), 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), 2),
            nn.ReLU(),
            nn.Conv2d(64, 32, (3, 3), 1),
            nn.Flatten(),
            nn.Linear(7 * 7 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, s: State) -> torch.Tensor:
        rlt = cast(torch.Tensor, self.network(
            s.to(DEVICE)))
        assert rlt.shape == (s.size(0), 1)
        return rlt.cpu()


class PPO(AlgorithmInterface[State, Action]):
    def __init__(self, n_actions: int, sigma: float = 0.2, c2: float = 0.01, gamma: float = 0.99):
        self.frame_skip = 0
        self.name = "ppo"
        self.n_actions = n_actions
        self.actor = Actor(n_actions).to(DEVICE)
        self.critic = Critic().to(DEVICE)
        self.times = -1

        self.gamma = gamma
        self.update_freq = 250
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), 1e-4)

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), 2.5e-4)

        self.sigma = sigma
        self.c2 = c2
        self.loss_func = torch.nn.MSELoss()

        self.loss = -1
        self.target = -1

    def allowed_actions(self, state: State) -> List[Action]:
        return list(range(self.n_actions))

    def take_action(self, state: State) -> ActionInfo[Action]:
        with torch.no_grad():

            act_probs = self.actor(resolve_lazy_frames(state))
            dist = Categorical(act_probs)
            act = dist.sample()

            return (cast(int, act.item()), {"log_prob": dist.log_prob(act)})

    def after_step(
        self,
        sar: Tuple[State, ActionInfo[Action], Reward],
        sa: Tuple[State, Optional[ActionInfo[Action]]],
    ):
        self.times += 1
        (s, a, r) = sar
        (sn, an) = sa

        self.memory.append((s, a, r, sn, an))

        if self.times != 0 and self.times % self.update_freq == 0:
            self.train()

        pass

    def train(self):
        (_, _, _, st, at) = self.memory[-1]
        if at is None:
            return

        all_states = [resolve_lazy_frames(s)
                      for (s, _, _, _, _) in self.memory]
        all_states.append(resolve_lazy_frames(st))

        all_states = torch.cat(all_states)
        all_states.shape == (len(self.memory) + 1, 4,  84, 84)

        actions = [a for (_, (a, _), _, _, _) in self.memory]
        actions.append(at[0])
        actions = torch.tensor(actions).unsqueeze(1)

        assert actions.shape == (len(self.memory) + 1, 1)

        log_probs = [i['log_prob'] for (_, (_, i), _, _, _) in self.memory]
        log_probs.append(at[1]['log_prob'])

        log_probs = torch.tensor(log_probs).unsqueeze(1)

        assert log_probs.shape == (len(self.memory) + 1, 1)

        predict_values = self.critic(all_states)

        assert predict_values.shape == (len(self.memory) + 1, 1)

        rets = cast(List[float], [])

        T = len(self.memory)

        v_st = self.critic(resolve_lazy_frames(st))
        for t, (_, _, _, _, _) in enumerate(self.memory):
            _ret = 0.0

            for j in range(t, T):
                (_, _, r, _, _) = self.memory[j]
                _ret += self.gamma ** (j - t) * r

            _ret += self.gamma ** (T - t) * v_st

            rets.append(_ret)

        rets.append(v_st)

        rets = torch.cat(cast(List[torch.Tensor], rets))
        assert rets.shape == (len(self.memory) + 1, 1)

        advs = (rets - predict_values).detach()

        assert advs.shape == (len(self.memory) + 1, 1)

        ratios = torch.exp(Categorical(
            self.actor(all_states)).log_prob(actions.squeeze(1)).unsqueeze(1) - log_probs)
        # ratios = [
        #     torch.exp(Categorical(self.actor(resolve_lazy_frames(s))).log_prob(
        #         torch.tensor(a)) - info["log_prob"])
        #     for (s, (a, info), _, _, _) in self.memory
        # ]

        # ratios.append(torch.exp(Categorical(self.actor(
        #     resolve_lazy_frames(st))).log_prob(torch.tensor(at[0])) - at[1]['log_prob']))

        # ratios = torch.stack(ratios)
        assert ratios.shape == (len(self.memory) + 1, 1)

        loss_clip = torch.min(
            ratios * advs, torch.clamp(ratios, 1 - self.sigma, 1 + self.sigma) * advs)

        assert loss_clip.shape == (len(self.memory) + 1, 1)

        loss_entropy = [Categorical(self.actor(resolve_lazy_frames(s))).entropy()
                        for (s, _, _, _, _) in self.memory]
        loss_entropy.append(Categorical(
            self.actor(resolve_lazy_frames(st))).entropy())

        loss_entropy = torch.stack(loss_entropy)
        assert loss_entropy.shape == (len(self.memory) + 1, 1)

        target = -loss_clip - self.c2 * loss_entropy
        assert target.shape == (len(self.memory) + 1, 1)

        self.actor_optimizer.zero_grad()
        self.target = target.mean()
        self.target.backward()
        self.actor_optimizer.step()

        loss_values = self.loss_func(predict_values, rets)

        self.critic_optimizer.zero_grad()
        self.loss = loss_values
        loss_values.backward()
        self.critic_optimizer.step()

        self.reset()

    def on_termination(
        self, sar: Tuple[List[State], List[ActionInfo[Action]], List[Reward]]
    ):
        (s, a, r) = sar
        assert len(s) == len(a) + 1
        assert len(s) == len(r) + 1
        pass

    def reset(self):
        self.memory: List[Transition] = []


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
        self.reset()

    def reset(self):
        pass

    def get_current_state(self, h: List[Observation]) -> State:
        assert len(h) > 0

        assert h[-1].shape == (4, 1, 84, 84)
        return h[-1]
