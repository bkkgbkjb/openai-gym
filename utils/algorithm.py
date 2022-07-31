from abc import abstractmethod, ABC
from typing import Callable, Dict, Generic, Literal, Tuple, TypeVar, List, Union, Any, Union, Optional
from utils.common import Action, Info, Reward, ActionInfo, S
from utils.transition import TransitionTuple

Mode = Union[Literal['train'], Literal['eval']]
ReportInfo = Info


class Algorithm(ABC, Generic[S]):

    def set_name(self, name: str):
        assert not hasattr(self, "name")
        self.name = name

    def on_agent_init(self, info: Dict[str, Any]):
        pass

    def on_agent_reset(self):
        pass

    def on_toggle_eval(self, isEval: bool):
        pass

    def on_env_reset(self, mode: Mode, info: Dict[str, Any]):
        pass

    @abstractmethod
    def take_action(self, mode: Mode, state: S, info: Info) -> Union[ActionInfo, Action]:
        raise NotImplementedError()

    @abstractmethod
    def manual_train(self, info: Dict[str, Any]):
        raise NotImplementedError()

    def after_step(self, mode: Mode, transition: TransitionTuple[S]):
        pass

    def on_episode_termination(
        self, mode: Mode, sari: Tuple[List[S], List[Action], List[Reward],
                                      List[Info]]
    ) -> Optional[ReportInfo]:
        pass

    def set_reporter(self, reporter: Callable[[Dict[str, Any]], None]):
        self.reporter = reporter

    def report(self, info: Dict[str, Any]):
        assert hasattr(self, "name")
        if self.reporter:
            self.reporter({(self.name + "/" + k): v
                           for (k, v) in info.items()})
