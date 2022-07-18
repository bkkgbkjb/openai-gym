import setup
from utils.algorithm import ActionInfo, Mode, ReportInfo
from utils.common import ActionScale, Info
from utils.env import expose_markers
from utils.episode import Episode
from utils.step import NotNoneStep, Step
from utils.transition import Transition, resolve_transitions, TransitionTuple
from torch import Tensor, nn
from collections import deque
import torch
from utils.preprocess import PreprocessI
from utils.algorithm import Algorithm
from torch.distributions import Categorical, Normal
from typing import Union
from utils.nets import NeuralNetworks, layer_init
from torch.utils.data import DataLoader
from args import args

from typing import List, Tuple, Any, Optional, Callable, Dict, cast
import numpy as np

from utils.replay_buffer import ReplayBuffer
from algms.bcq.algorithm import BCQ

State = torch.Tensor
O = torch.Tensor
Action = torch.Tensor

S = O
Reward = int

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Goal = torch.Tensor


class Preprocess(PreprocessI[O, S]):
    def __init__(self):
        pass

    def on_agent_reset(self):
        pass

    def get_current_state(self, h: List[O]) -> S:
        assert len(h) > 0

        # assert h[-1].shape == (4, 1, 84, 84)
        return torch.from_numpy(h[-1]).type(torch.float32).to(DEVICE)


class SAEncoder(NeuralNetworks):
    def __init__(self, n_state: int, n_action: int, repr_dim: int = 16):
        super().__init__()
        self.n_state = n_state
        self.n_action = n_action
        self.n_repr = repr_dim

        self.net = nn.Sequential(
            layer_init(nn.Linear(self.n_state + self.n_action, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, self.n_repr)),
        )

    def forward(self, state: State, action: Action):
        return self.net(torch.cat([state, action], dim=1))


class GEncoder(NeuralNetworks):
    def __init__(self, n_goal: int, repr_dim: int = 16):
        super().__init__()
        self.n_goal = n_goal
        self.n_repr = repr_dim

        self.net = nn.Sequential(
            layer_init(nn.Linear(self.n_goal, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, self.n_repr)),
        )

    def forward(self, state: State):
        return self.net(state)


class QFunction(NeuralNetworks):
    def __init__(self, n_state: int, n_action: int, n_goal: int):
        super().__init__()
        self.sa_encoder = SAEncoder(n_state, n_action)
        self.g_encoder = GEncoder(n_goal)

        self.n_state = n_state
        self.n_action = n_action
        self.n_goal = n_goal

    def forward(
        self, s: State, a: Action, goal: Goal
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        sa_value = self.sa_encoder(s, a)
        g_value = self.g_encoder(goal)
        return torch.einsum("bi,bi->b", sa_value, g_value), (sa_value, g_value)


class PaiFunction(NeuralNetworks):
    def __init__(self, n_state: int, n_action: int, n_goal: int, action_scale: float):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(n_state + n_goal, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.ReLU(),
        ).to(DEVICE)

        self.mean = layer_init(nn.Linear(1024, n_action)).to(DEVICE)
        self.std = layer_init(nn.Linear(1024, n_action)).to(DEVICE)

        self.n_state = n_state
        self.n_action = n_action
        self.action_scale = action_scale
        self.n_goal = n_goal

    def forward(self, s: State, g: Goal) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(torch.cat([s.to(DEVICE), g.to(DEVICE)], dim=1))
        mean = self.mean(x)
        log_std = self.std(x)
        log_std = torch.clamp(log_std, min=-12, max=6)

        assert mean.shape == (s.size(0), self.n_action)
        assert log_std.shape == (s.size(0), self.n_action)

        return mean, log_std

    def sample(self, s: State, g: Goal):
        mean, log_std = self.forward(s, g)
        std = log_std.exp()
        normal = Normal(mean, std)
        raw_act = normal.rsample()
        assert raw_act.shape == (s.size(0), self.n_action)

        y_t = torch.tanh(raw_act)
        act = y_t * self.action_scale

        raw_log_prob = normal.log_prob(raw_act)
        assert raw_log_prob.shape == (s.size(0), self.n_action)

        mod_log_prob = (self.action_scale * (1 - y_t.pow(2)) + 1e-6).log()
        assert mod_log_prob.shape == (s.size(0), self.n_action)

        log_prob = (raw_log_prob - mod_log_prob).sum(1, keepdim=True)

        mean = self.action_scale * torch.tanh(mean)
        return act, log_prob, mean, normal


class H(Algorithm):
    def __init__(
        self,
        n_state: int,
        n_goals: int,
        n_actions: int,
        action_scale: ActionScale,
    ):
        self.set_name("H")
        self.n_actions = n_actions
        self.n_state = n_state
        self.n_goals = n_goals

        self.gamma = 0.99
        self.action_scale = action_scale

        self.high_level = None

        self.c = 25

        self.reset()

    def reset(self):
        self.episodes = None
        self.high_replay_memory = ReplayBuffer[Transition](None)
        self.reset_episode_info()

    def reset_episode_info(self):
        self.inner_steps = 0
        self.fg = None
        self.add_marker = None
        self.del_markers = None

    def on_env_reset(self, mode: Mode, info: Dict[str, Any]):
        assert mode == "eval"

        assert self.fg is None
        env = info["env"]

        self.fg = torch.tensor(env.target_goal, dtype=torch.float32, device=DEVICE)

        assert self.add_marker is None
        assert self.del_markers is None

        self.add_marker, self.del_markers = expose_markers(env.unwrapped.viewer)

    @torch.no_grad()
    def take_action(self, mode: Mode, state: State) -> Action:
        assert self.fg is not None
        action, _, max_actions, _ = self.policy.sample(
            state.unsqueeze(0), self.fg.unsqueeze(0)
        )

        return (max_actions if mode == "eval" else action).squeeze()

    def after_step(self, mode: Mode, transition: TransitionTuple[S]):
        assert mode == "eval"

        if self.inner_steps % 5 == 0:
            assert self.del_markers is not None
            self.del_markers()
            assert self.add_marker is not None
            self.add_marker((0.5, 0.5, 0.5), f"{self.inner_steps}")

        self.inner_steps += 1

    def on_episode_termination(
        self, mode: Mode, sari: Tuple[List[S], List[Action], List[Reward], List[Info]]
    ) -> Optional[ReportInfo]:
        (_, _, r, _) = sari
        report = {"success_rate": 1 if any([_r >= 0.7 for _r in r]) else 0}

        self.reset_episode_info()
        return report

    def train(self):
        assert self.high_level is not None
        self.high_level.manual_train(dict(transitions=self.high_replay_memory))

    def get_episodes(self, episodes: List[Episode]):
        action_scale = [-float("inf")] * self.n_state

        for e in episodes:
            for se in Episode.cut(e, self.c):
                assert se.len == self.c
                se.compute_returns(self.gamma)

                act = se.last_state - se[0].state
                assert len(act.shape) == 1

                for i, a in enumerate(act):
                    if a > action_scale[i]:
                        action_scale[i] = a

                self.high_replay_memory.append(
                    Transition(
                        (
                            NotNoneStep(
                                se[0].state,
                                act,
                                se[0].info["return"],
                            ),
                            Step(se.last_state, None, None, dict(end=se.end)),
                        )
                    )
                )
        
        self.action_scale = torch.Tensor(action_scale)
        self.high_level = BCQ(self.n_state, self.n_state, self.action_scale)

    def manual_train(self, info: Dict[str, Any]):
        assert "episodes" in info
        if self.episodes != info["episodes"]:
            assert self.episodes is None
            self.episodes = info["episodes"]
            self.get_episodes(self.episodes)

        self.train()
        return self.high_level.batch_size
