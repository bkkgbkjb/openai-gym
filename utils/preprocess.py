from abc import abstractmethod
from typing import Protocol, TypeVar, List, Union
from utils.env_sb3 import LazyFrames, resolve_lazy_frames
import torch

O = TypeVar("O")
S = TypeVar('S', bound=Union[torch.Tensor, LazyFrames], covariant=True)


class Preprocess(Protocol[O, S]):

    @abstractmethod
    def get_current_state(self, h: List[O]) -> S:
        raise NotImplementedError()

    @abstractmethod
    def on_agent_reset(self):
        raise NotImplementedError()
