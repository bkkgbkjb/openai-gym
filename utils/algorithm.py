from abc import abstractmethod
from optparse import Option
from typing import Optional, Protocol, Tuple, TypeVar, List, Union
import numpy as np
from utils.common import Episode


S = TypeVar("S", contravariant=True)
A = TypeVar("A")
R = float


class AlgorithmInterface(Protocol[S, A]):

    frame_skip: int

    @abstractmethod
    def allowed_actions(self, state: S) -> List[A]:
        raise NotImplementedError()

    @abstractmethod
    def take_action(self, state: S) -> A:
        raise NotImplementedError()

    @abstractmethod
    def after_step(self,  sar: Tuple[S, A, R], sa: Tuple[S, Optional[A]]):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()
