from abc import abstractmethod
from typing import Protocol, TypeVar, List

from utils.common import AllowedState

O = TypeVar("O")
S = TypeVar("S", covariant=True, bound=AllowedState)


class PreprocessInterface(Protocol[O, S]):

    @abstractmethod
    def get_current_state(self, h: List[O]) -> S:
        raise NotImplementedError()

    @abstractmethod
    def on_agent_reset(self):
        raise NotImplementedError()
