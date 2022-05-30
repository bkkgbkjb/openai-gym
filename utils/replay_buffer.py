from collections import deque
from typing import Deque, Generic, Optional, TypeVar, Tuple, List, cast, Union
import numpy as np
import torch

from utils.common import AllowedState, Transition
from utils.env_sb3 import LazyFrames, resolve_lazy_frames

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SARSA = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
              torch.Tensor]

S = TypeVar("S", bound=AllowedState)


class ReplayBuffer(Generic[S]):

    def __init__(self,
                 state_shape: Tuple,
                 action_shape: Tuple,
                 capacity: int = int(1e6)):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)
        self.state_shape = state_shape
        self.action_shape = action_shape

    def append(self, transition: Transition):
        assert self.state_shape == transition[0].shape
        assert self.action_shape == transition[1][0].shape
        assert self.state_shape == transition[3].shape
        if transition[4] is not None and isinstance(transition[4], tuple):
            assert transition[4][0].shape == self.action_shape

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

    def sample(self, size: int) -> List[Transition]:
        assert self.len > 0
        idx = np.random.choice(len(self.buffer), size)
        l = list(self.buffer)

        r: List[Transition] = []
        for i in idx:
            r.append(l[i])

        return r

    @staticmethod
    def resolve(mini_batch: List[Transition], state_shape: Tuple,
                action_shape: Tuple) -> SARSA:

        l = len(mini_batch)
        states = torch.stack([
            s if isinstance(s, torch.Tensor) else resolve_lazy_frames(s)
            for (s, _, _, _, _) in mini_batch
        ])
        assert states.shape == ((l, ) + state_shape)

        actions = torch.stack([
            torch.from_numpy(a).type(torch.float32)
            if a.dtype == np.float64 else torch.from_numpy(a)
            for (_, (a, _), _, _, _) in mini_batch
        ])

        assert actions.shape == ((l, ) + action_shape)

        rewards = torch.stack([
            torch.tensor(r, dtype=torch.float32)
            for (_, _, r, _, _) in mini_batch
        ]).unsqueeze(1)

        assert rewards.shape == (l, 1)

        next_states = torch.stack([
            sn if isinstance(sn, torch.Tensor) else resolve_lazy_frames(sn)
            for (_, _, _, sn, _) in mini_batch
        ])

        assert next_states.shape == ((l, ) + state_shape)

        done = torch.as_tensor(
            [
                1 if (an is None or an == True) else 0
                for (_, _, _, _, an) in mini_batch
            ],
            dtype=torch.int8,
        ).unsqueeze(1)

        assert done.shape == (l, 1)

        return (
            states.to(DEVICE),
            actions.to(DEVICE),
            rewards.to(DEVICE),
            next_states.to(DEVICE),
            done.to(DEVICE),
        )
