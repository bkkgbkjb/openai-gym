import numpy as np
import torch
import gym
from utils.env_sb3 import LazyFrames, resolve_lazy_frames
from typing import (
    Any,
    TypeVar,
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
from utils.algorithm import Algorithm
from utils.preprocess import Preprocess
from utils.common import ActionInfo, Action, Reward
from torch.utils.data import DataLoader

R = Reward

AS = TypeVar('AS', bound=Union[torch.Tensor, LazyFrames])
AO = TypeVar("AO")


class Agent(Generic[AO, AS]):

    def __init__(
        self,
        env: gym.Env[AO, Action],
        algm: Algorithm[AS],
        preprocess: Preprocess[AO, AS],
    ):
        self.env = env
        self.algm = algm
        self.preprocess = preprocess
        self.name: str = algm.name
        self.algm.on_init({'env': self.env})
        self.reset()

    def get_current_state(self, obs_episode: List[AO]) -> AS:
        state = self.preprocess.get_current_state(obs_episode)
        assert isinstance(state, torch.Tensor) or isinstance(
            state,
            LazyFrames), "preprocess.get_current_state应该返回tensor或lazyframes"
        return state

    def reset(self, ):
        self.ready_act: Optional[ActionInfo] = None
        self.end = False

        self.observation_episode: List[AO] = []
        self.state_episode: List[AS] = []
        self.action_episode: List[ActionInfo] = []
        self.reward_episode: List[R] = []

        self.eval_observation_episode: List[AO] = []

        assert len(self.state_episode) == len(self.observation_episode)

        self.preprocess.on_agent_reset()
        self.algm.on_agent_reset()

    def set_algm_reporter(self, reporter: Callable[[Dict[str, Any]], None]):
        self.report = reporter
        self.algm.set_reporter(reporter)

    def toggleEval(self, newEval: bool):
        self.algm.on_toggle_eval(newEval)

    def format_action(
            self, a: Union[Action,
                           ActionInfo]) -> Tuple[Action, Dict[str, Any]]:
        if isinstance(a, tuple):
            return a
        return (a, dict())

    def eval(self, env: gym.Env) -> Tuple[float, Tuple[List[AO]]]:
        assert len(self.eval_observation_episode) == 0
        assert not self.end, "should reset before eval agnet"
        self.toggleEval(True)

        o = cast(AO, env.reset())
        self.eval_observation_episode.append(o)

        s = False
        rwd = 0.0

        while not s:
            actinfo = self.get_action(
                self.get_current_state(self.eval_observation_episode))
            act = actinfo[0]

            (o, r, s,
             _) = env.step(act[0] if isinstance(env.action_space, gym.spaces.
                                                Discrete) else act)
            rwd += r
            self.eval_observation_episode.append(o)

        self.report({'eval_return': rwd})
        return rwd, (self.eval_observation_episode, )

    def get_action(self, state: AS) -> ActionInfo:
        actinfo = self.format_action(self.algm.take_action(state))
        act = actinfo[0]

        action_space = self.env.action_space

        assert isinstance(action_space, gym.spaces.Discrete) or isinstance(
            action_space, gym.spaces.Box), "目前只能处理Discrete和Box两种action_space"

        if isinstance(action_space, gym.spaces.Discrete):
            assert act.shape == (1, )

        if isinstance(action_space, gym.spaces.Box):
            assert act.shape == action_space.shape

        return actinfo

    def train(
        self
    ) -> Tuple[float, Tuple[List[AO], List[AS], List[ActionInfo], List[R]]]:
        assert not self.end, "agent needs to be reset before training"
        self.toggleEval(False)

        o = cast(AO, self.env.reset())
        self.observation_episode.append(o)
        self.state_episode.append(
            self.get_current_state(self.observation_episode))

        stop = False

        while not stop:

            actinfo = self.ready_act or self.get_action(self.state_episode[-1])

            act = actinfo[0]
            (obs, rwd, stop, _) = self.env.step(act[0] if isinstance(
                self.env.action_space, gym.spaces.Discrete) else act)

            self.action_episode.append(actinfo)
            self.reward_episode.append(rwd)

            obs = cast(AO, obs)
            self.observation_episode.append(obs)
            self.state_episode.append(
                self.get_current_state(self.observation_episode))

            self.ready_act = (None if stop else self.get_action(
                self.state_episode[-1]))

            assert len(self.state_episode) == len(self.observation_episode)

            self.algm.after_step((
                self.state_episode[-2],
                self.action_episode[-1],
                self.reward_episode[-1],
                self.state_episode[-1],
                self.ready_act,
            ))

        assert len(self.state_episode) == len(self.observation_episode) == len(
            self.action_episode) + 1 == len(self.reward_episode) + 1

        self.end = True

        self.algm.on_episode_termination(
            (self.state_episode, self.action_episode, self.reward_episode))

        total_rwd = np.sum([r for r in self.reward_episode])

        self.report({"train_return": total_rwd})

        return total_rwd, (self.observation_episode, self.state_episode,
                           self.action_episode, self.reward_episode)

    def close(self):
        self.reset()
        self.env.close()


OS = TypeVar('OS', bound=Union[torch.Tensor, LazyFrames])
OO = TypeVar("OO")


class OfflineAgent(Generic[OO, OS]):

    def __init__(self, dataloader: DataLoader, algm: Algorithm[OS],
                 preprocess: Preprocess[OO, OS]):
        self.dataloader = dataloader
        self.algm = algm
        self.preprocess = preprocess
        self.name: str = algm.name
        self.algm.on_init({'dataloader': dataloader})
        self.reset()

        self.data_iter = iter(self.dataloader)

    def reset(self):
        self.preprocess.on_agent_reset()
        self.algm.on_agent_reset()
        self.eval_observation_episode: List[OO] = []

    def set_algm_reporter(self, reporter: Callable[[Dict[str, Any]], None]):
        self.report = reporter
        self.algm.set_reporter(reporter)

    def toggleEval(self, newEval: bool):
        self.algm.on_toggle_eval(newEval)

    def train(self) -> int:
        self.toggleEval(False)
        return self.algm.manual_train()

    def format_action(
            self, a: Union[Action,
                           ActionInfo]) -> Tuple[Action, Dict[str, Any]]:
        if isinstance(a, tuple):
            return a
        return (a, dict())

    def get_current_state(self, obs_episode: List[OO]) -> OS:
        state = self.preprocess.get_current_state(obs_episode)
        assert isinstance(state, torch.Tensor) or isinstance(
            state,
            LazyFrames), "preprocess.get_current_state应该返回tensor或lazyframes"
        return state

    def get_action(self, state: OS) -> ActionInfo:
        actinfo = self.format_action(self.algm.take_action(state))
        act = actinfo[0]

        action_space = self.env.action_space

        assert isinstance(action_space, gym.spaces.Discrete) or isinstance(
            action_space, gym.spaces.Box), "目前只能处理Discrete和Box两种action_space"

        if isinstance(action_space, gym.spaces.Discrete):
            assert act.shape == (1, )

        if isinstance(action_space, gym.spaces.Box):
            assert act.shape == action_space.shape

        return actinfo

    def eval(self, env: gym.Env) -> Tuple[float, Tuple[List[OO]]]:
        assert len(self.eval_observation_episode) == 0
        self.env = env
        self.toggleEval(True)

        o = cast(OO, env.reset())
        self.eval_observation_episode.append(o)

        s = False
        rwd = 0.0

        while not s:
            actinfo = self.get_action(
                self.get_current_state(self.eval_observation_episode))
            act = actinfo[0]

            (o, r, s,
             _) = env.step(act[0] if isinstance(env.action_space, gym.spaces.
                                                Discrete) else act)
            rwd += r
            self.eval_observation_episode.append(cast(OO, o))

        self.report({'eval_return': rwd})
        return rwd, (self.eval_observation_episode, )


AAO = TypeVar('AAO')
AAS = TypeVar('AAS', bound=Union[torch.Tensor, LazyFrames])
AllAgent = Union[Agent[AAO, AAS], OfflineAgent[AAO, AAS]]
