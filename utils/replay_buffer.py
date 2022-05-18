from collections import deque
from typing import Deque, Generic, Optional, TypeVar, Tuple, List, cast
import numpy as np
import torch

from utils.common import TransitionGeneric
from utils.env_sb3 import LazyFrames, resolve_lazy_frames

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SARSA = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
              torch.Tensor]

S = TypeVar("S")


class ReplayBuffer(Generic[S]):

    def __init__(self, capacity: int = int(1e6)):
        self.buffer: Deque[TransitionGeneric[S]] = deque(maxlen=capacity)

    def append(self, transition: TransitionGeneric[S]):
        self.buffer.append(transition)

        return self

    def clear(self):
        self.buffer.clear()
        return self

    @property
    def size(self) -> int:
        assert self.buffer.maxlen is not None
        return self.buffer.maxlen

    @property
    def len(self) -> int:
        return len(self.buffer)

    def sample(self, size: int) -> List[TransitionGeneric[S]]:
        idx = np.random.choice(len(self.buffer), size)
        l = list(self.buffer)

        r: List[TransitionGeneric[S]] = []
        for i in idx:
            r.append(l[i])

        return r

    @staticmethod
    def resolve(mini_batch: List[TransitionGeneric[S]]) -> SARSA:

        states = torch.stack([
            s if isinstance(s, torch.Tensor) else resolve_lazy_frames(
                cast(LazyFrames, s)) for (s, _, _, _, _) in mini_batch
        ])

        actions = torch.stack([
            torch.from_numpy(a).type(torch.float32)
            if a.dtype == np.float64 else torch.from_numpy(a)
            for (_, (a, _), _, _, _) in mini_batch
        ])

        rewards = torch.stack([
            torch.tensor(r, dtype=torch.float32)
            for (_, _, r, _, _) in mini_batch
        ]).unsqueeze(1)

        next_states = torch.stack([
            sn if isinstance(sn, torch.Tensor) else resolve_lazy_frames(
                cast(LazyFrames, sn)) for (_, _, _, sn, _) in mini_batch
        ])

        done = torch.as_tensor(
            [1 if an is None else 0 for (_, _, _, _, an) in mini_batch],
            dtype=torch.int8,
        ).unsqueeze(1)

        return (
            states.to(DEVICE),
            actions.to(DEVICE),
            rewards.to(DEVICE),
            next_states.to(DEVICE),
            done.to(DEVICE),
        )
