from abc import abstractmethod
from optparse import Option
from typing import Callable, Dict, Optional, Protocol, Tuple, TypeVar, List, Union, Any
import numpy as np

from utils.common import ActionInfo


S = TypeVar("S")
A = TypeVar("A")
R = float


class AlgorithmInterface(Protocol[S, A]):

    name: str

    @abstractmethod
    def take_action(self, state: S) -> Union[ActionInfo[A], A]:
        raise NotImplementedError()

    @abstractmethod
    def after_step(
        self, sar: Tuple[S, ActionInfo[A], R], sa: Tuple[S, Optional[ActionInfo[A]]]
    ):
        raise NotImplementedError()

    @abstractmethod
    def on_episode_termination(self, sar: Tuple[List[S], List[ActionInfo[A]], List[R]]):
        raise NotImplementedError()

    @abstractmethod
    def on_agent_reset(self):
        raise NotImplementedError()

    def set_reporter(self, reporter: Callable[[Dict[str, Any]], None]):
        raise NotImplementedError()
