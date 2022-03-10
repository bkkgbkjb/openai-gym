import setup
from utils.common import Step, Episode, TransitionGeneric
from torch import nn
import math
from collections import deque
import torch
from utils.preprocess import PreprocessInterface
from utils.algorithm import AlgorithmInterface
import plotly.graph_objects as go
from tqdm.autonotebook import tqdm
from torchvision import transforms as T
from utils.agent import Agent
from gym.spaces import Box
from typing import List, Tuple, Literal, Any, Optional, cast, Callable, Union, Iterable
import gym
import numpy.typing as npt
from utils.env import PreprocessObservation, FrameStack
import numpy as np

Observation = torch.Tensor
Action = int

State = torch.Tensor
Reward = int

Transition = TransitionGeneric[State, Action]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
