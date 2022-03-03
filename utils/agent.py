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
        self.ready_act: Optional[A] = None
        self.end = False

        self.episode_observation: List[O] = []
        self.episode_action: List[A] = []
        self.episode_reward: List[float] = []
        self.episode_state: List[S] = []

        o: O = self.env.reset()
        self.episode_observation.append(o)
        self.episode_state.append(
            self.preprocess.get_current_state(self.episode_observation)
        )

        if comps == "all" or "preprocess" in comps:
            self.preprocess.reset()
        if comps == "all" or "algorithm" in comps:
            self.algm.reset()

    def toggleImprove(self, newImprov: bool):
        self.improv = newImprov

    def step(self) -> Tuple[O, bool, Optional[Episode[O, A]]]:
        assert not self.end, "cannot step on a ended agent"

        act = self.ready_act or self.algm.take_action(
            self.preprocess.get_current_state(self.episode_observation)
        )

        rwd: float = 0.0
        obs: Optional[O] = None
        stop: bool = False

        for _ in range(self.algm.frame_skip + 1):
            (o, r, s, _) = self.env.step(act)
            rwd += r
            obs = o
            stop = s
            if stop:
                break

        # self.episode[-1] = (self.episode[-1][0], act, rwd)
        self.episode_action.append(act)
        self.episode_reward.append(rwd)

        obs = cast(O, obs)
        # self.episode.append((obs, None, None))
        self.episode_observation.append(obs)
        self.episode_state.append(
            self.preprocess.get_current_state(self.episode_observation)
        )

        self.ready_act = (
            None if stop else (self.algm.take_action(self.episode_state[-1]))
        )

        self.improv and self.algm.after_step(
            (
                self.episode_state[-1],
                self.ready_act,
            ),
            # self.preprocess.transform_history(self.episode_observation[:-1]),
            (self.episode_state[-2], self.episode_action[-2], self.episode_reward[-2]),
        )

        if not stop:
            return (obs, stop, None)

        self.end = True

        self.improv and self.algm.on_termination(
            self.preprocess.transform_history(self.episode_observation[:-1])
        )
        return (obs, stop, None)

    def render(self, mode: str):
        self.env.render(mode)

    def close(self):
        self.env.close()
        self.reset()
