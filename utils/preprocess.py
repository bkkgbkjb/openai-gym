from abc import abstractmethod
from typing import Optional, Protocol, TypeVar, List, Tuple


S = TypeVar('S')
O = TypeVar('O')
A = TypeVar('A')
R = float


class PreprocessInterface(Protocol[O, A, S]):
    @abstractmethod
    def transform_one(self, h: List[Tuple[O, Optional[A], Optional[R]]]) -> S:
        raise NotImplementedError()

    @abstractmethod
    def transform_many(self, h: List[Tuple[O, Optional[A], Optional[R]]]) -> List[Tuple[S, Optional[A], Optional[R]]]:
        raise NotImplementedError()
