from abc import abstractmethod
from typing import Protocol, TypeVar, List

from utils.common import AllowedState as S

O = TypeVar("O")


class Preprocess(Protocol[O]):

    @abstractmethod
    def get_current_state(self, h: List[O]) -> S:
        raise NotImplementedError()

    @abstractmethod
    def on_agent_reset(self):
        raise NotImplementedError()
