from abc import abstractmethod
from typing import Callable, Dict, Generic, Tuple, TypeVar, List, Union, Any, Union
from utils.common import Action, Info, Reward
from utils.env_sb3 import LazyFrames
from utils.transition import TransitionTuple

import torch

S = TypeVar('S', bound=Union[torch.Tensor, LazyFrames])
ActionInfo = Tuple[Action, Dict[str, Any]]


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
    
    def on_env_reset(self, info: Dict[str, Any]):
        pass

    def after_step(self, transition: TransitionTuple[S]):
        pass

    def on_episode_termination(self, sari: Tuple[List[S], List[Action],
                                                 List[Reward], List[Info]]):
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
