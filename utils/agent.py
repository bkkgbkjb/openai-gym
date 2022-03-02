from abc import abstractmethod
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

    def reset(self, comps: Union[List[Literal["preprocess", "algorithm"]], Literal['all']] = []):
        self.ready_act: Optional[A] = None
        self.end = False
        self.episode: Episode[O, A] = []

        o: O = self.env.reset()
        self.episode.append((o, None, None))
        if comps == 'all' or 'preprocess' in comps:
            self.preprocess.reset()
        if comps == 'all' or 'algorithm' in comps:
            self.algm.reset()

    def toggleImprove(self, newImprov: bool):
        self.improv = newImprov

    def step(self) -> Tuple[O, bool, Optional[Episode[O, A]]]:
        assert not self.end, "cannot step on a ended agent"

        act = self.ready_act or self.algm.take_action(
            self.preprocess.get_current_state(self.episode)
        )
        (obs, rwd, stop, _) = self.env.step(act)
        obs = cast(O, obs)

        assert len(self.episode) >= 1

        self.episode[-1] = (self.episode[-1][0], act, rwd)


        self.episode.append((obs, None, None))

        self.ready_act = self.algm.take_action(
            self.preprocess.get_current_state(self.episode)
        )

        self.improv and self.algm.after_step(
            (self.preprocess.get_current_state(self.episode), self.ready_act),
            self.preprocess.transform_history(self.episode[:-1]),
        )

        if stop:
            # self.episodes.append(self.episode)
            self.end = True
            # self.episode.append((self.cur_obs, None, None))
            self.improv and self.algm.on_termination(
                self.preprocess.transform_history(self.episode)
            )
            # self.episode = []
            return (obs, stop, self.episode)

        return (obs, stop, None)

    def render(self, mode: str):
        self.env.render(mode)

    def close(self):
        self.env.close()
        self.reset()
