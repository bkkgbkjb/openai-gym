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
        self.ready_act: Optional[Tuple[A, Dict[str, Any]]] = None
        self.end = False

        self.observation_episode: List[O] = []
        self.state_episode: List[S] = []
        self.action_episode: List[Tuple[A, Dict[str, Any]]] = []
        self.reward_episode: List[R] = []

        o: O = self.env.reset()
        self.observation_episode.append(o)
        self.state_episode.append(
            self.preprocess.get_current_state(self.observation_episode)
        )
        assert len(self.state_episode) == len(self.observation_episode)

        self.preprocess.on_agent_reset()
        self.algm.on_agent_reset()

    def set_algm_reporter(self, reporter: Callable[[Dict[str, Any]], None]):
        self.algm.set_reporter(reporter)

    def toggleEval(self, newEval: bool):
        self.eval = newEval

    def format_action(self, a: Union[A, ActionInfo[A]]) -> Tuple[A, Dict[str, Any]]:
        if isinstance(a, tuple):
            return a
        return (a, dict())

    def step(self) -> Tuple[O, bool]:
        assert not self.end, "cannot step on a ended agent"

        act = self.ready_act or self.format_action(
            self.algm.take_action(self.state_episode[-1])
        )

        (obs, rwd, stop, _) = self.env.step(act[0])

        self.action_episode.append(act)
        self.reward_episode.append(rwd)

        obs = cast(O, obs)
        self.observation_episode.append(obs)
        self.state_episode.append(
            self.preprocess.get_current_state(self.observation_episode)
        )

        self.ready_act = (
            None
            if stop
            else self.format_action(self.algm.take_action(self.state_episode[-1]))
        )

        assert len(self.state_episode) == len(self.observation_episode)

        self.eval or self.algm.after_step(
            (self.state_episode[-2], self.action_episode[-1],
             self.reward_episode[-1]),
            (
                self.state_episode[-1],
                self.ready_act,
            ),
        )

        if stop:
            assert len(self.action_episode) == len(self.reward_episode)
            assert len(self.state_episode) == len(self.action_episode) + 1

            self.end = True

            self.eval or self.algm.on_episode_termination(
                (self.state_episode, self.action_episode, self.reward_episode)
            )

        return (obs, stop)

    @property
    def steps(self):
        return len(self.action_episode)


    def render(self, mode: str):
        self.env.render(mode)

    def close(self):
        self.env.close()
        self.reset()
