from abc import ABC
from typing import Generic, Tuple, TypeVar, Optional, cast

from utils.common import Info, Action, Reward, AllAllowedStates, AllowedStates
import numpy as np
import torch

from utils.env_sb3 import LazyFrames

S = TypeVar('S', bound=AllAllowedStates)
A = TypeVar('A')
R = TypeVar('R')


class BaseStep(ABC, Generic[S, A, R]):

    def __init__(self, s: S, a: A, r: R, i: Info):
        self.s = s
        self.a = a
        self.r = r
        self.i = i

        assert isinstance(self.s, torch.Tensor) or isinstance(
            self.s, LazyFrames)
        assert isinstance(i, dict)
        assert 'end' in self.i
        assert isinstance(self.i['end'], bool)

    def is_end(self) -> bool:
        return self.i['end']

    @property
    def details(self) -> Tuple[S, A, R, Info]:
        return (self.s, self.a, self.r, self.i)

    @property
    def state(self) -> S:
        return self.s

    @property
    def action(self) -> A:
        return self.a

    @property
    def reward(self) -> R:
        return self.r

    @property
    def info(self) -> Info:
        return self.i


NNS = TypeVar('NNS', bound=AllowedStates)

SS = TypeVar('SS', bound=AllowedStates)


class Step(BaseStep[SS, Optional[Action], Optional[Reward]], Generic[SS]):

    def __init__(self, s: SS, a: Optional[Action], r: Optional[Reward],
                 i: Optional[Info]):
        super().__init__(s, a, r, dict(end=False) if i is None else i)

        assert self.a is None or isinstance(self.a, np.ndarray)
        assert r is None or isinstance(r, float)

    def is_not_none(self) -> bool:
        return self.a is not None and self.r is not None


class NotNoneStep(BaseStep[NNS, Action, Reward], Generic[NNS]):

    def __init__(self, s: NNS, a: Action, r: Reward, i: Info):
        super().__init__(s, a, r, i)

    def to_step(self) -> Step[NNS]:
        return cast(Step[NNS], self)