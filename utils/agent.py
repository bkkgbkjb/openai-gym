from abc import abstractmethod
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


Action = TypeVar("Action")
State = TypeVar("State")
Observation = TypeVar("Observation")

StateAction = Tuple[State, Action]
Reward = float
Step = Tuple[Observation, Optional[Action], Optional[Reward]]
Episode = List[Step]

# Episode = List


class PreprocessInterface(Protocol):
    @abstractmethod
    def transform(o: Observation) -> State:
        raise NotImplementedError()


class Agent(Generic[Observation, Action]):
    def __init__(
        self,
        env: gym.Env,
        algm: AlgorithmInterface,
    ):
        self.env = env
        self.algm = algm
        self.reset()

    def reset(self):
        self.cur_obs: Observation = self.env.reset()
        self.ready_act: Optional[Action] = None
        self.end = False
        self.episode: Episode = []

    def step(self) -> Tuple[Observation, bool, Optional[Episode]]:
        assert not self.end, "cannot step on a ended agent"

        act = self.ready_act or self.algm.take_action(self.cur_obs, self.omega)
        (obs, rwd, stop, _) = self.env.step(act)
        obs = cast(Observation, obs)

        self.episode.append((self.cur_obs, act, rwd))

        self.cur_obs = obs

        self.ready_act = self.algm.take_action(self.cur_obs)

        self.algm.after_step((self.cur_obs, self.ready_act), self.episode)

        if stop:
            # self.episodes.append(self.episode)
            self.end = True
            self.episode.append((self.cur_obs, None, None))
            self.algm.on_termination(self.episode)
            # self.episode = []
            return (obs, stop, self.episode)

        return (obs, stop, None)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
        self.clear()

    def predict(self, s: Observation) -> float:
        return np.max([self.algm.predict((s, a)) for a in self.algm.allowed_actions(s)])
