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
from algms.cql.algorithm import CQL_SAC

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

        self.c = 50

        self.reset()

    def reset(self):
        self.episodes = None
        self.high_replay_memory = ReplayBuffer[Transition](None)
        self.reset_episode_info()

    def reset_episode_info(self):
        self.inner_steps = 0
        self.action_space = None
        self.fg = None
        self.sub_goal = None
        self.add_marker = None
        self.del_markers = None

    def on_env_reset(self, mode: Mode, info: Dict[str, Any]):
        assert mode == "eval"

        assert self.fg is None
        env = info["env"]
        assert self.action_space is None
        self.action_space = env.action_space

        self.fg = torch.tensor(env.target_goal, dtype=torch.float32, device=DEVICE)

        assert self.add_marker is None
        assert self.del_markers is None

        self.add_marker, self.del_markers = expose_markers(env.unwrapped.viewer)

    @torch.no_grad()
    def take_action(self, mode: Mode, state: State) -> Action:
        assert self.fg is not None
        assert self.high_level is not None

        if self.inner_steps % self.c == 0:
            self.sub_goal = self.high_level.take_action(mode, torch.cat([state, self.fg])) + state

            assert self.del_markers is not None
            self.del_markers()

            assert self.add_marker is not None
            x, y = self.sub_goal.cpu().detach().numpy()[:2]
            self.add_marker((x, y, 0.5), f"{self.inner_steps}")

        assert self.action_space is not None

        if self.inner_steps != 0 and self.inner_steps % self.c == self.c - 1:
            x, y = self.sub_goal.cpu().detach().numpy()[:2]
            return torch.tensor([x, y] + [-999] * (self.n_actions -2 ))

        return torch.from_numpy(self.action_space.sample()).type(torch.float32) * 0.5
    
    def transition_subgoal(self, s: torch.Tensor, sn: torch.Tensor):
        assert self.sub_goal is not None
        self.sub_goal = s + self.sub_goal - sn


    def after_step(self, mode: Mode, transition: TransitionTuple[S]):
        (s1, s2) = transition
        assert mode == "eval"


        self.transition_subgoal(s1.state, s2.state)
        self.inner_steps += 1

    def on_episode_termination(
        self, mode: Mode, sari: Tuple[List[S], List[Action], List[Reward], List[Info]]
    ) -> Optional[ReportInfo]:
        (_, _, r, _) = sari
        report = {"success_rate": 1 if any([_r >= 0.7 for _r in r]) else 0}

        assert self.del_markers is not None
        self.del_markers()

        self.reset_episode_info()
        return report

    def train(self):
        assert self.high_level is not None
        self.high_level.manual_train(dict(transitions=self.high_replay_memory.as_list()))

    def get_episodes(self, episodes: List[Episode]):
        action_scale = [-float("inf")] * self.n_state

        for e in episodes:
            for se in Episode.cut(e, self.c, start=np.random.choice(self.c), allow_last_not_align=True):
                # assert se.len == self.c
                se.compute_returns(self.gamma)

                act = se.last_state - se[0].state
                assert len(act.shape) == 1

                for i, a in enumerate(act):
                    if a > action_scale[i]:
                        action_scale[i] = a

                goal = torch.from_numpy(se[0].info['goal']).type(torch.float32)
                self.high_replay_memory.append(
                    Transition(
                        (
                            NotNoneStep(
                                torch.cat([se[0].state, goal]),
                                act,
                                se[0].info["return"],
                            ),
                            Step(
                                torch.cat([se.last_state, goal]),
                                None,
                                None,
                                dict(end=se.end),
                            ),
                        )
                    )
                )

        self.action_scale = torch.Tensor(action_scale).to(DEVICE)
        self.high_level = CQL_SAC(self.n_state + self.n_goals, self.n_state, self.action_scale)
        self.high_level.set_reporter(self.reporter)

    def manual_train(self, info: Dict[str, Any]):
        assert "episodes" in info
        if self.episodes != info["episodes"]:
            assert self.episodes is None
            self.episodes = info["episodes"]
            self.get_episodes(self.episodes)

        assert self.high_level is not None
        self.train()
        return self.high_level.batch_size
