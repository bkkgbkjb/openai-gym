from abc import abstractmethod
from optparse import Option
from typing import Optional, Protocol, Tuple, TypeVar, List
import numpy as np


S = TypeVar('S')
A = TypeVar('A')
R = float
Episode = List[Tuple[S, Optional[A], Optional[R]]]


class AlgorithmInterface(Protocol[S, A]):

    @abstractmethod
    def allowed_actions(self, state: S) -> List[A]:
        raise NotImplementedError()

    @abstractmethod
    def take_action(self, state: S) -> A:
        raise NotImplementedError()

    @abstractmethod
    def after_step(self, sa: Tuple[S, A], episode: List[Tuple[S, Optional[A], Optional[R]]]):
        raise NotImplementedError()

    @abstractmethod
    def on_termination(self, episode: List[Tuple[S, Optional[A], Optional[R]]]):
        raise NotImplementedError()
