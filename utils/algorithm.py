from abc import abstractmethod
from optparse import Option
from typing import Optional, Protocol, Tuple, TypeVar, List, Union
import numpy as np
from utils.common import Episode


S = TypeVar("S")
A = TypeVar("A")
# Episode = List[Tuple[S, Optional[A], Optional[R]]]


class AlgorithmInterface(Protocol[S, A]):
    

    @abstractmethod
    def allowed_actions(self, state: S) -> List[A]:
        raise NotImplementedError()

    @abstractmethod
    def take_action(self, state: S) -> Union[A, Tuple[A, int]]:
        raise NotImplementedError()

    @abstractmethod
    def after_step(self, sa: Tuple[S, Optional[A]], episode: Episode[S, A]):
        raise NotImplementedError()
    
    after_step_freq: int

    @abstractmethod
    def on_termination(self, episode: Episode[S, A]):
        raise NotImplementedError()
    
    need_on_termination: bool

    @abstractmethod
    def reset(self):
        raise NotImplementedError()
