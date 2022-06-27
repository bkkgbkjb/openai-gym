import setup
from utils.algorithm import ActionInfo, Mode
from utils.common import Info, Reward, Action
from utils.transition import (
    TransitionTuple,
)
import torch
from utils.algorithm import Algorithm

from typing import List, Tuple, Any, Optional, Callable, Dict
from papers.sac import NewSAC
import numpy as np

MAX_TIMESTEPS = 500
ACTION_SCALE = 1.0

Observation = torch.Tensor

State = Observation
Goal = torch.Tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class HighNetwork(Algorithm):
    def __init__(self, state_dim: int, goal_dim: int, action_dim: int) -> None:
        self.set_name('high_network')

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.goal_dim = goal_dim

        self.sac = NewSAC(
            self.state_dim + self.goal_dim, self.goal_dim, 20.0, True, False
        )

        self.random_episode = 300
        self.reset()

    def set_reporter(self, reporter: Callable[[Dict[str, Any]], None]):
        self.sac.set_reporter(reporter)

    def reset(self):
        self.times = 0
        self.epoch = 0
        self.eval = False
    
    def on_toggle_eval(self, isEval: bool):
        self.eval = isEval
        self.sac.on_toggle_eval(isEval)

    @torch.no_grad()
    def take_action(self, s: State, dg: Goal) -> Action:
        if self.eval:
            assert self.sac.eval
            obs = torch.cat([s, dg])
            return self.sac.take_action(obs)

        act = None
        if self.epoch <= self.random_episode:
            act = torch.from_numpy( np.random.uniform(-20, 20, self.goal_dim)).type(torch.float32).to(DEVICE)

        else:
            assert not self.sac.eval
            obs = torch.cat([s, dg])
            assert obs.shape == (self.state_dim + self.goal_dim,)
            act = self.sac.take_action(obs)

        assert act.shape == (self.goal_dim,)
        return act

    def after_step(self, mode: Mode, transition: TransitionTuple[State]):
        self.sac.after_step(mode, transition)
        # if mode == 'train' and self.sac.replay_memory.len >= 128:
        #     self.sac.train()

        self.times += 1

    def on_episode_termination_train(
        self, sari: Tuple[List[State], List[Action], List[Reward], List[Info]]
    ):
        self.epoch += 1
