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
from utils.algorithm import Algorithm, ActionInfo
from utils.preprocess import PreprocessI
from utils.common import Action, Info, Reward
from torch.utils.data import DataLoader

from utils.step import NotNoneStep, Step

R = Reward

AS = TypeVar("AS", bound=Union[torch.Tensor, LazyFrames])
AO = TypeVar("AO")


class Agent(Generic[AO, AS]):
    def __init__(
        self,
        env: gym.Env,
        algm: Algorithm[AS],
        preprocess: PreprocessI[AO, AS],
    ):
        self.env = env
        self.algm = algm
        assert self.algm.name, "agent必须有一个名称(name)"
        self.preprocess = preprocess
        self.name: str = algm.name
        self.algm.on_init({"env": self.env})
        self.reset()

    def get_current_state(self, obs_episode: List[AO]) -> AS:
        state = self.preprocess.get_current_state(obs_episode)
        assert isinstance(state, torch.Tensor) or isinstance(
            state, LazyFrames
        ), "preprocess.get_current_state应该返回tensor或lazyframes"
        return state

    def reset(
        self,
    ):
        self.end = False

        self.observation_episode: List[AO] = []
        self.state_episode: List[AS] = []
        self.action_episode: List[Action] = []
        self.reward_episode: List[R] = []
        self.info_episode: List[Info] = []

        self.eval_observation_episode: List[AO] = []

        assert len(self.state_episode) == len(self.observation_episode)

        self.preprocess.on_agent_reset()
        self.algm.on_agent_reset()

    def set_algm_reporter(self, reporter: Callable[[Dict[str, Any]], None]):
        self.report = reporter
        self.algm.set_reporter(reporter)

    def toggleEval(self, newEval: bool):
        self.algm.on_toggle_eval(newEval)

    def format_action(self, a: Union[Action, ActionInfo]) -> ActionInfo:
        if isinstance(a, tuple):
            assert "end" not in a[1]
            assert isinstance(a[0], torch.Tensor)
            a[1]["end"] = False
            return a
        
        assert isinstance(a, torch.Tensor)
        return (a, dict(end=False))

    def format_env_reset(self, env: gym.Env) -> AO:
        o = env.reset()

        if not isinstance(o, tuple):
            self.algm.on_env_reset(dict())
            return o

        assert len(o) == 2
        assert isinstance(o[1], dict)
        self.algm.on_env_reset(o[1])
        return cast(AO, o[0])

    def eval(self, env: gym.Env) -> Tuple[float, Tuple[List[AO]]]:
        assert len(self.eval_observation_episode) == 0
        assert not self.end, "should reset before eval agnet"
        self.toggleEval(True)

        o = self.format_env_reset(env)

        self.eval_observation_episode.append(o)

        s = False
        rwd = 0.0

        while not s:
            actinfo = self.get_action(
                self.get_current_state(self.eval_observation_episode)
            )
            act = actinfo[0]

            (o, r, s, _) = env.step(
                act[0].numpy() if isinstance(env.action_space, gym.spaces.Discrete) else act.numpy()
            )
            rwd += r
            self.eval_observation_episode.append(o)

        self.report({"eval_return": rwd})
        return rwd, (self.eval_observation_episode,)

    def get_action(self, state: AS) -> ActionInfo:
        actinfo = self.format_action(self.algm.take_action(state))
        act = actinfo[0]

        action_space = self.env.action_space

        assert isinstance(action_space, gym.spaces.Discrete) or isinstance(
            action_space, gym.spaces.Box
        ), "目前只能处理Discrete和Box两种action_space"

        if isinstance(action_space, gym.spaces.Discrete):
            assert act.shape == (1,)

        if isinstance(action_space, gym.spaces.Box):
            assert act.shape == action_space.shape

        return actinfo

    def train(
        self,
    ) -> Tuple[float, Tuple[List[AO], List[AS], List[Action], List[R], List[Info]]]:
        assert not self.end, "agent needs to be reset before training"
        self.toggleEval(False)

        o = self.format_env_reset(self.env)
        self.observation_episode.append(o)
        self.state_episode.append(self.get_current_state(self.observation_episode))

        stop = False

        while not stop:

            actinfo = self.get_action(self.state_episode[-1])

            act, info = actinfo
            (obs, rwd, stop, env_info) = self.env.step(
                act[0].numpy()
                if isinstance(self.env.action_space, gym.spaces.Discrete)
                else act.numpy()
            )

            info["env_info"] = env_info

            self.action_episode.append(act)
            self.info_episode.append(info)
            self.reward_episode.append(rwd)

            obs = cast(AO, obs)
            self.observation_episode.append(obs)
            self.state_episode.append(self.get_current_state(self.observation_episode))

            assert len(self.state_episode) == len(self.observation_episode)

            self.algm.after_step(
                (
                    NotNoneStep(
                        self.state_episode[-2],
                        self.action_episode[-1],
                        self.reward_episode[-1],
                        self.info_episode[-1],
                    ),
                    Step(self.state_episode[-1], None, None, dict(end=stop)),
                )
            )

        self.info_episode.append(dict(end=True))
        assert (
            len(self.state_episode)
            == len(self.observation_episode)
            == len(self.action_episode) + 1
            == len(self.reward_episode) + 1
            == len(self.info_episode)
        )

        self.end = True

        self.algm.on_episode_termination(
            (
                self.state_episode,
                self.action_episode,
                self.reward_episode,
                self.info_episode,
            )
        )

        total_rwd = np.sum([r for r in self.reward_episode])

        self.report({"train_return": total_rwd})

        return total_rwd, (
            self.observation_episode,
            self.state_episode,
            self.action_episode,
            self.reward_episode,
            self.info_episode,
        )

    def close(self):
        self.reset()
        self.env.close()


OS = TypeVar("OS", bound=Union[torch.Tensor, LazyFrames])
OO = TypeVar("OO")


class OfflineAgent(Generic[OO, OS]):
    def __init__(
        self,
        dataloader: DataLoader,
        algm: Algorithm[OS],
        preprocess: PreprocessI[OO, OS],
    ):
        self.dataloader = dataloader
        self.algm = algm
        self.preprocess = preprocess
        self.name: str = algm.name
        self.algm.on_init({"dataloader": dataloader})
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
        self, a: Union[Action, ActionInfo]
    ) -> Tuple[Action, Dict[str, Any]]:
        if isinstance(a, tuple):
            return a
        return (a, dict())

    def get_current_state(self, obs_episode: List[OO]) -> OS:
        state = self.preprocess.get_current_state(obs_episode)
        assert isinstance(state, torch.Tensor) or isinstance(
            state, LazyFrames
        ), "preprocess.get_current_state应该返回tensor或lazyframes"
        return state

    def get_action(self, state: OS) -> ActionInfo:
        actinfo = self.format_action(self.algm.take_action(state))
        act = actinfo[0]

        action_space = self.env.action_space

        assert isinstance(action_space, gym.spaces.Discrete) or isinstance(
            action_space, gym.spaces.Box
        ), "目前只能处理Discrete和Box两种action_space"

        if isinstance(action_space, gym.spaces.Discrete):
            assert act.shape == (1,)

        if isinstance(action_space, gym.spaces.Box):
            assert act.shape == action_space.shape

        return actinfo

    def format_env_reset(self, env: gym.Env) -> OO:
        o = env.reset()

        if not isinstance(o, tuple):
            self.algm.on_env_reset(dict())
            return o

        assert len(o) == 2
        assert isinstance(o[1], dict)
        self.algm.on_env_reset(o[1])
        return o[0]

    def eval(self, env: gym.Env) -> Tuple[float, Tuple[List[OO]]]:
        assert len(self.eval_observation_episode) == 0
        self.env = env
        self.toggleEval(True)

        # o = cast(OO, env.reset())
        o = self.format_env_reset(env)
        self.eval_observation_episode.append(o)

        s = False
        rwd = 0.0

        while not s:
            actinfo = self.get_action(
                self.get_current_state(self.eval_observation_episode)
            )
            act = actinfo[0]

            (o, r, s, _) = env.step(
                act[0].numpy() if isinstance(env.action_space, gym.spaces.Discrete) else act.numpy()
            )
            rwd += r
            self.eval_observation_episode.append(cast(OO, o))

        self.report({"eval_return": rwd})
        return rwd, (self.eval_observation_episode,)


AAO = TypeVar("AAO")
AAS = TypeVar("AAS", bound=Union[torch.Tensor, LazyFrames])
AllAgent = Union[Agent[AAO, AAS], OfflineAgent[AAO, AAS]]
