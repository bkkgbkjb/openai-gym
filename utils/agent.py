from abc import abstractmethod
from argparse import Action
from optparse import Option
import numpy as np
import gym
from typing import (
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
from utils.common import Step, Episode


A = TypeVar("A")
S = TypeVar("S")
O = TypeVar("O")


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
        self.improv = True
        self.reset()

    def reset(
        self,
        comps: Union[List[Literal["preprocess", "algorithm"]], Literal["all"]] = [],
    ):
        self.ready_act: Optional[Tuple[A, int]] = None
        self.end = False
        self.episode: Episode[O, A] = []

        o: O = self.env.reset()
        self.episode.append((o, None, None))

        if comps == "all" or "preprocess" in comps:
            self.preprocess.reset()
        if comps == "all" or "algorithm" in comps:
            self.algm.reset()

        self.times = 0

    def toggleImprove(self, newImprov: bool):
        self.improv = newImprov

    def format_act(self, act: Union[A, Tuple[A, int]]) -> Tuple[A, int]:
        if isinstance(act, tuple):
            assert act[1] > 0
            return (cast(A, act[0]), act[1])
        else:
            return (act, 1)

    def get_act(self) -> Union[None, Tuple[A, int]]:
        if self.ready_act is None:
            return None

        assert self.ready_act[1] >= 0
        if self.ready_act[1] > 0:
            return self.ready_act

        return None

    def need_after_step(self) -> bool:
        return (
            self.improv
            and self.algm.after_step_freq > 0
            and self.times % self.algm.after_step_freq == 0
        )

    def need_on_termination(self) -> bool:
        return (
            self.improv
            and self.algm.need_on_termination
        )

    def step(self) -> Tuple[O, bool, Optional[Episode[O, A]]]:
        assert not self.end, "cannot step on a ended agent"

        act = self.get_act() or self.format_act(
            self.algm.take_action(self.preprocess.get_current_state(self.episode))
        )

        (obs, rwd, stop, _) = self.env.step(act[0])

        self.ready_act = (act[0], act[1] - 1)

        self.episode[-1] = (self.episode[-1][0], act[0], rwd)

        obs = cast(O, obs)
        self.episode.append((obs, None, None))

        self.ready_act = (
            None
            if stop
            else (
                self.get_act()
                or self.format_act(
                    self.algm.take_action(
                        self.preprocess.get_current_state(self.episode)
                    )
                )
            )
        )

        self.need_after_step() and self.algm.after_step(
            (
                self.preprocess.get_current_state(self.episode),
                self.ready_act and self.ready_act[0],
            ),
            self.preprocess.transform_history(self.episode[:-1]),
        )

        if not stop:
            self.times += 1
            return (obs, stop, None)

        self.end = True
        self.need_on_termination() and self.algm.on_termination(
            self.preprocess.transform_history(self.episode)
        )
        self.times += 1
        return (obs, stop, self.episode)

    def render(self, mode: str):
        self.env.render(mode)

    def close(self):
        self.env.close()
        self.reset()
