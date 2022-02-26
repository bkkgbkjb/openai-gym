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


A = TypeVar("A")
S = TypeVar("S")
O = TypeVar("O")


Reward = float


class Agent(Generic[O, S, A]):

    def __init__(
        self,
        env: gym.Env,
        algm: AlgorithmInterface[S, A],
        preprocess: PreprocessInterface[O, A, S]
    ):
        self.env = env
        self.algm = algm
        self.preprocess = preprocess
        self.reset()

    def reset(self):
        # self.cur_obs: O = self.env.reset()
        self.ready_act: Optional[A] = None
        self.end = False
        self.episode: List[Tuple[O, Optional[A], Optional[Reward]]] = []

        o: O = self.env.reset()
        self.episode.append((o, None, None))

    def step(self) -> Tuple[O, bool, Optional[List[Tuple[O, Optional[A], Optional[Reward]]]]]:
        assert not self.end, "cannot step on a ended agent"

        act = self.ready_act or self.algm.take_action(
            self.preprocess.transform_one(self.episode))
        (obs, rwd, stop, _) = self.env.step(act)
        obs = cast(O, obs)

        # self.episode.append((self.cur_obs, act, rwd))
        assert len(self.episode) >= 1

        self.episode[-1] = (self.episode[-1][0], act, rwd)

        # self.cur_obs = obs

        self.episode.append((obs, None, None))

        self.ready_act = self.algm.take_action(
            self.preprocess.transform_one(self.episode))

        self.algm.after_step(
            (self.preprocess.transform_one(self.episode), self.ready_act), self.preprocess.transform_many(self.episode))

        if stop:
            # self.episodes.append(self.episode)
            self.end = True
            # self.episode.append((self.cur_obs, None, None))
            self.algm.on_termination(
                self.preprocess.transform_many(self.episode))
            # self.episode = []
            return (obs, stop, self.episode)

        return (obs, stop, None)

    def render(self, mode: str):
        self.env.render(mode)

    def close(self):
        self.env.close()
        self.reset()
