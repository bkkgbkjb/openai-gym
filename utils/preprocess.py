from abc import abstractmethod
from typing import Protocol, TypeVar, List, Union, Tuple
from utils.common import Action, Info, LazyFrames, Reward, resolve_lazy_frames
import torch

O = TypeVar("O", covariant=True)
S = TypeVar('S', covariant=True)

AllInfo = Tuple[List[O], List[S], List[Action], List[Reward], List[Info]]


class PreprocessI(Protocol[O, S]):

    @abstractmethod
    def get_current_state(self, osari: AllInfo) -> S:
        raise NotImplementedError()

    def on_agent_reset(self):
        pass
