from abc import abstractmethod
from optparse import Option
from typing import Callable, Dict, Optional, Protocol, Tuple, TypeVar, List, Union, Any
import numpy as np
from pytz import NonExistentTimeError

from utils.common import ActionInfo


S = TypeVar("S")
A = TypeVar("A")
R = float


class AlgorithmInterface(Protocol[S, A]):

    frame_skip: int
    name: str

    @abstractmethod
    def allowed_actions(self, state: S) -> List[A]:
        raise NotImplementedError()

    @abstractmethod
    def take_action(self, state: S) -> Union[ActionInfo[A], A]:
        raise NotImplementedError()

    @abstractmethod
    def after_step(
        self, sar: Tuple[S, ActionInfo[A], R], sa: Tuple[S, Optional[ActionInfo[A]]]
    ):
        raise NotImplementedError()

    @abstractmethod
    def on_termination(self, sar: Tuple[List[S], List[ActionInfo[A]], List[R]]):
        raise NotImplementedError()

    @abstractmethod
    def on_reset(self):
        raise NotImplementedError()

    @abstractmethod
    def set_reporter(self, reporter: Callable[[Dict[Any, Any]], None]):
        raise NotImplementedError()
