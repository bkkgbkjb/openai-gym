from abc import abstractmethod
from typing import Optional, Protocol, TypeVar, List, Tuple
from utils.common import Episode

S = TypeVar('S')
O = TypeVar('O')
A = TypeVar('A')


class PreprocessInterface(Protocol[O, A, S]):
    @abstractmethod
    def transform_one(self, h: Episode[O, A]) -> S:
        raise NotImplementedError()

    @abstractmethod
    def transform_many(self, h: Episode[O, A]) -> Episode[S, A]:
        raise NotImplementedError()
