from collections import deque
from typing import Deque, Generic, Optional, TypeVar, Tuple, List, cast, Union
import numpy as np
import torch

from utils.common import AllowedState, Transition
from utils.env_sb3 import LazyFrames, resolve_lazy_frames

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SARSA = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
              torch.Tensor]


class ReplayBuffer:

    def __init__(self,
                 state_shape: Tuple,
                 action_shape: Tuple,
                 capacity: Optional[int] = int(1e6)):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)
        self.state_shape = state_shape
        self.action_shape = action_shape

    def append(self, transition: Transition):
        self.buffer.append(transition)

        return self

    def clear(self):
        self.buffer.clear()
        return self

    @property
    def size(self) -> Optional[int]:
        return self.buffer.maxlen

    @property
    def len(self) -> int:
        return len(self.buffer)

    def sample(self, size: int) -> List[Transition]:
        assert self.len > 0
        idx = np.random.choice(len(self.buffer), size)

        r: List[Transition] = []
        for i in idx:
            r.append(self.buffer[i])

        return r
