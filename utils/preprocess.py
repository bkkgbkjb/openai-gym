from abc import abstractmethod
from typing import Optional, Protocol, TypeVar, List, Tuple
from utils.common import Episode

S = TypeVar('S')
O = TypeVar('O')
A = TypeVar('A')


class PreprocessInterface(Protocol[O, A, S]):
    @abstractmethod
    def get_current_state(self, h: Episode[O, A]) -> S:
        raise NotImplementedError()

    @abstractmethod
    def transform_history(self, h: Episode[O, A]) -> Episode[S, A]:
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()
