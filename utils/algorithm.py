from abc import abstractmethod
from typing import Callable, Dict, Optional, Protocol, Tuple, TypeVar, List, Union, Any, Union
from utils.common import Action, ActionInfo, AllowedState as S, NotNoneStep, Reward, Transition


class Algorithm:

    name: str

    @abstractmethod
    def take_action(self, state: S) -> Union[ActionInfo, Action]:
        raise NotImplementedError()

    @abstractmethod
    def after_step(self, transition: Transition):
        raise NotImplementedError()

    @abstractmethod
    def on_episode_termination(self, sar: Tuple[List[S], List[ActionInfo],
                                                List[Reward]]):
        raise NotImplementedError()

    @abstractmethod
    def on_agent_reset(self):
        raise NotImplementedError()

    @abstractmethod
    def on_toggle_eval(self, isEval: bool):
        raise NotImplementedError()

    def set_reporter(self, reporter: Callable[[Dict[str, Any]], None]):
        pass
