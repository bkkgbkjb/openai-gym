from abc import abstractmethod
from typing import Optional, Protocol, TypeVar, List, Tuple
from utils.common import Episode

S = TypeVar("S", covariant=True)
O = TypeVar("O")
A = TypeVar("A", covariant=True)


class PreprocessInterface(Protocol[O, A, S]):
    @abstractmethod
    def get_current_state(self, h: List[O]) -> S:
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()
