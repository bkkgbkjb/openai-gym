from collections import deque
from typing import Deque, Generic, Optional, TypeVar, Tuple, List, cast
import numpy as np
import torch

from utils.common import TransitionGeneric

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SARSA = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


S = TypeVar("S")
A = TypeVar("A")


class ReplayBuffer(Generic[S, A]):
    def __init__(self, capacity: int = int(1e6)):
        self.buffer: Deque[TransitionGeneric[S, A]] = deque(maxlen=capacity)

    def append(self, transition: TransitionGeneric[S, A]):
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

    def sample(self, size: int) -> List[TransitionGeneric[S, A]]:
        idx = np.random.choice(len(self.buffer), size)
        l = list(self.buffer)

        r: List[TransitionGeneric[S, A]] = []
        for i in idx:
            r.append(l[i])

        return r

    @staticmethod
    def resolve(
        mini_batch: List[TransitionGeneric[S, A]],
    ) -> SARSA:

        states = torch.stack([cast(torch.Tensor, s) for (s, _, _, _, _) in mini_batch])

        actions = torch.stack(
            [
                torch.from_numpy(a).type(torch.float32)
                for (_, (a, _), _, _, _) in mini_batch
            ]
        )

        rewards = torch.stack(
            [torch.tensor(r, dtype=torch.float32) for (_, _, r, _, _) in mini_batch]
        ).unsqueeze(1)

        next_states = torch.stack(
            [cast(torch.Tensor, sn) for (_, _, _, sn, _) in mini_batch]
        )

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
