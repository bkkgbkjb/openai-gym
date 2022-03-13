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
        self.ready_act: Optional[A] = None
        self.end = False

        self.episode_observation: List[O] = []
        self.episode_state: List[S] = []
        self.episode_action: List[A] = []
        self.episode_reward: List[R] = []

        o: O = self.env.reset()
        self.episode_observation.append(o)
        self.episode_state.append(
            self.preprocess.get_current_state(self.episode_observation)
        )

        self.preprocess.on_reset()
        self.algm.on_reset()

    def toggleEval(self, newEval: bool):
        self.eval = newEval

    def step(self) -> Tuple[O, bool]:
        assert not self.end, "cannot step on a ended agent"

        act = self.ready_act or self.algm.take_action(self.episode_state[-1])

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

        self.episode_action.append(act)
        self.episode_reward.append(rwd)

        obs = cast(O, obs)
        self.episode_observation.append(obs)
        self.episode_state.append(
            self.preprocess.get_current_state(self.episode_observation)
        )

        self.ready_act = (
            None if stop else (self.algm.take_action(self.episode_state[-1]))
        )

        self.eval or self.algm.after_step(
            (self.episode_state[-2], self.episode_action[-1], self.episode_reward[-1]),
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
