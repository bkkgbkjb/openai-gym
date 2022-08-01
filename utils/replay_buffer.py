from collections import deque
from typing import Deque, Generic, Optional, TypeVar, Tuple, List, cast, Union
import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SARSA = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
              torch.Tensor]

E = TypeVar("E")


class ReplayBuffer(Generic[E]):

    def __init__(
            self,
            capacity: Optional[int] = int(1e6),
    ):
        self.buffer: Deque[E] = deque(maxlen=capacity)
    
    def as_list(self) -> List[E]:
        return list(self.buffer)

    def append(self, e: E):
        self.buffer.append(e)

        return self

    def details(self) -> List[E]:
        return list(self.buffer)

    def clear(self):
        self.buffer.clear()
        return self

    @property
    def size(self) -> Optional[int]:
        return self.buffer.maxlen

    @property
    def len(self) -> int:
        return len(self.buffer)

    def sample(self, size: int, p: Optional[List[float]] = None) -> List[E]:
        assert self.len > 0
        idx = np.random.choice(len(self.buffer), size, p=p)

        r: List[E] = []
        for i in idx:
            r.append(self.buffer[i])

        return r
