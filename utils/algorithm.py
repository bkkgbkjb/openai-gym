from abc import abstractmethod
from typing import Callable, Dict, Generic, Optional, Protocol, Tuple, TypeVar, List, Union, Any, Union
from utils.common import Action, ActionInfo, AllowedStates, NotNoneStep, Reward, Transition, TransitionTuple
from utils.env_sb3 import LazyFrames, resolve_lazy_frames
import torch

S = TypeVar('S', bound=Union[torch.Tensor, LazyFrames])


class Algorithm(Generic[S]):

    name: str

    def on_init(self, info: Dict[str, Any]):
        pass

    @abstractmethod
    def take_action(self, state: S) -> Union[ActionInfo, Action]:
        raise NotImplementedError()

    @abstractmethod
    def manual_train(self):
        raise NotImplementedError()

    def after_step(self, transition: TransitionTuple[S]):
        pass

    def on_episode_termination(self, sar: Tuple[List[S], List[ActionInfo],
                                                List[Reward]]):
        pass

    def on_agent_reset(self):
        pass

    def on_toggle_eval(self, isEval: bool):
        pass

    def set_reporter(self, reporter: Callable[[Dict[str, Any]], None]):
        self.reporter = reporter

    def report(self, info: Dict[str, Any]):
        if self.reporter:
            self.reporter(info)
