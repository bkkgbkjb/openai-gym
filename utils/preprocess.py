from abc import abstractmethod
from typing import Protocol, TypeVar, List

O = TypeVar("O")
S = TypeVar("S", covariant=True)


class PreprocessInterface(Protocol[O, S]):
    @abstractmethod
    def get_current_state(self, h: List[O]) -> S:
        raise NotImplementedError()

    @abstractmethod
    def on_agent_reset(self):
        raise NotImplementedError()
