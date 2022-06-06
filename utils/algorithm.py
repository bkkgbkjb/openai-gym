from abc import abstractmethod
from typing import Callable, Dict, Optional, Protocol, Tuple, TypeVar, List, Union, Any, Union
from utils.common import Action, ActionInfo, AllowedState as S, NotNoneStep, Reward, Transition


class Algorithm:

    name: str

    def on_init(self, info: Dict[str, Any]):
        pass

    @abstractmethod
    def take_action(self, state: S) -> Union[ActionInfo, Action]:
        raise NotImplementedError()

    @abstractmethod
    def manual_train(self):
        raise NotImplementedError()

    def after_step(self, transition: Transition):
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
