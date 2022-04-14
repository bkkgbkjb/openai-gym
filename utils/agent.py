from abc import abstractmethod
from argparse import Action
from optparse import Option
import numpy as np
import gym
from typing import (
    Any,
    Generic,
    Literal,
    List,
    Tuple,
    cast,
    Dict,
    Optional,
    Callable,
    Protocol,
    Union,
    TypeVar,
)
from utils.algorithm import AlgorithmInterface
from utils.preprocess import PreprocessInterface
from utils.common import ActionInfo, StepGeneric, Episode


A = TypeVar("A")
S = TypeVar("S")
O = TypeVar("O")
R = float


class Agent(Generic[O, S, A]):
    def __init__(
        self,
        env: gym.Env,
        algm: AlgorithmInterface[S, A],
        preprocess: PreprocessInterface[O, A, S],
    ):
        self.env = env
        self.algm = algm
        self.preprocess = preprocess
        self.eval = False
        self.name: str = algm.name
        self.reset()

    def reset(
        self,
    ):
        self.ready_act: Optional[Tuple[A, Any]] = None
        self.end = False

        self.episode_observation: List[O] = []
        self.episode_state: List[S] = []
        self.episode_action: List[Tuple[A, Any]] = []
        self.episode_reward: List[R] = []

        o: O = self.env.reset()
        self.episode_observation.append(o)
        self.episode_state.append(
            self.preprocess.get_current_state(self.episode_observation)
        )

        self.preprocess.on_reset()
        self.algm.on_reset()

    def set_algm_reporter(self, reporter: Callable[[Dict[Any, Any]], None]):
        self.algm.set_reporter(reporter)

    def toggleEval(self, newEval: bool):
        self.eval = newEval

    def format_action(self, a: Union[A, ActionInfo[A]]) -> Tuple[A, Any]:
        if isinstance(a, tuple):
            return a
        return (a, dict())

    def step(self) -> Tuple[O, bool]:
        assert not self.end, "cannot step on a ended agent"

        act = self.ready_act or self.format_action(
            self.algm.take_action(self.episode_state[-1])
        )

        (obs, rwd, stop, _) = self.env.step(act[0])

        self.episode_action.append(act)
        self.episode_reward.append(rwd)

        obs = cast(O, obs)
        self.episode_observation.append(obs)
        self.episode_state.append(
            self.preprocess.get_current_state(self.episode_observation)
        )

        self.ready_act = (
            None
            if stop
            else self.format_action(self.algm.take_action(self.episode_state[-1]))
        )

        self.eval or self.algm.after_step(
            (self.episode_state[-2], self.episode_action[-1],
             self.episode_reward[-1]),
            (
                self.episode_state[-1],
                self.ready_act,
            ),
        )

        if stop:
            self.end = True
            self.eval or self.algm.on_termination(
                (self.episode_state, self.episode_action, self.episode_reward)
            )

        return (obs, stop)

    def render(self, mode: str):
        self.env.render(mode)

    def close(self):
        self.env.close()
        self.reset()
