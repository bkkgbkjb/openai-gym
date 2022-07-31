import numpy as np
import torch
import gym
from utils.common import LazyFrames, resolve_lazy_frames
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
from utils.algorithm import Algorithm, ActionInfo, Mode, ReportInfo
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
        algm: Algorithm[AS],
        preprocess: PreprocessI[AO, AS],
    ):
        self.algm = algm
        assert self.algm.name, "agent必须有一个名称(name)"
        self.preprocess = preprocess
        self.name: str = algm.name
        self.algm.on_agent_init({})
        self.reset()

    def reset(self, ):

        self.preprocess.on_agent_reset()
        self.algm.on_agent_reset()

    def toggleEval(self, newEval: bool):
        self.algm.on_toggle_eval(newEval)

    def set_algm_reporter(self, reporter: Callable[[Dict[str, Any]], None]):
        self.report = reporter
        self.algm.set_reporter(reporter)

    def get_current_state(self, obs_episode: List[AO]) -> AS:
        state = self.preprocess.get_current_state(obs_episode)
        assert isinstance(state, torch.Tensor) or isinstance(
            state,
            LazyFrames), "preprocess.get_current_state应该返回tensor或lazyframes"
        return state

    def format_action(self, a: Union[Action, ActionInfo]) -> ActionInfo:
        if isinstance(a, tuple):
            assert "end" not in a[1]
            assert isinstance(a[0], torch.Tensor)
            a[1]["end"] = False
            return (a[0].detach(), a[1])

        assert isinstance(a, torch.Tensor)
        return (a.detach(), dict(end=False))

    def get_action(self, state: AS, env: gym.Env, mode: Mode, info: Info) -> ActionInfo:
        actinfo = self.format_action(self.algm.take_action(mode, state, info))
        act = actinfo[0]

        action_space = env.action_space

        assert isinstance(action_space, gym.spaces.Discrete) or isinstance(
            action_space, gym.spaces.Box), "目前只能处理Discrete和Box两种action_space"

        if isinstance(action_space, gym.spaces.Discrete):
            assert act.shape == (1, )

        if isinstance(action_space, gym.spaces.Box):
            assert act.shape == action_space.shape

        return actinfo

    def reset_env(self, env: gym.Env, mode: Mode) -> AO:
        o = env.reset()

        if not isinstance(o, tuple):
            self.algm.on_env_reset(mode, dict(env=env, mode=mode))
            return o

        assert len(o) == 2
        assert isinstance(o[1], dict)

        assert 'env' not in o[1]
        o[1]['env'] = env
        assert 'mode' not in o[1]
        o[1]['mode'] = mode

        self.algm.on_env_reset(mode, o[1])
        return cast(AO, o[0])

    def train(self, env: gym.Env):
        return self.run(env, 'train')

    def eval(self, env: gym.Env):
        return self.run(env, 'eval')

    def run(
        self, env: gym.Env, mode: Union[Literal['eval'], Literal['train']]
    ) -> Tuple[ReportInfo, Tuple[List[AO], List[AS], List[Action], List[R],
                                 List[Info]]]:
        assert mode == 'train' or mode == 'eval'
        self.toggleEval(mode == 'eval')

        observation_episode: List[AO] = []
        state_episode: List[AS] = []
        action_episode: List[Action] = []
        reward_episode: List[R] = []
        info_episode: List[Info] = []

        o = self.reset_env(env, mode)
        observation_episode.append(o)
        state_episode.append(self.get_current_state(observation_episode))

        stop = False

        while not stop:

            actinfo = self.get_action(state_episode[-1], env, mode, dict(obs=observation_episode,states=state_episode,actions=action_episode,rewards=reward_episode,infos=info_episode))

            act, info = actinfo
            (obs, rwd, stop,
             env_info) = env.step(act[0].cpu().numpy() if isinstance(
                 env.action_space, gym.spaces.Discrete) else act.cpu().numpy())

            info["env_info"] = env_info

            action_episode.append(act)
            info_episode.append(info)
            reward_episode.append(rwd)

            obs = cast(AO, obs)
            observation_episode.append(obs)
            state_episode.append(self.get_current_state(observation_episode))

            assert len(state_episode) == len(observation_episode)

            self.algm.after_step(mode, (
                NotNoneStep(
                    state_episode[-2],
                    action_episode[-1],
                    reward_episode[-1],
                    info_episode[-1],
                ),
                Step(state_episode[-1], None, None, dict(end=stop)),
            ))

        info_episode.append(dict(end=True))
        assert (len(state_episode) == len(observation_episode) ==
                len(action_episode) + 1 == len(reward_episode) + 1 ==
                len(info_episode))

        report_info = self.algm.on_episode_termination(mode, (
            state_episode,
            action_episode,
            reward_episode,
            info_episode,
        )) or dict()

        total_rwd = np.sum([r for r in reward_episode])

        if mode == 'train':
            self.report({f"train/return": total_rwd})

        env.close()

        report_info['return'] = total_rwd

        return report_info, (
            observation_episode,
            state_episode,
            action_episode,
            reward_episode,
            info_episode,
        )


OS = TypeVar("OS", bound=Union[torch.Tensor, LazyFrames])
OO = TypeVar("OO")


class OfflineAgent(Generic[OO, OS]):

    def __init__(
        self,
        algm: Algorithm[OS],
        preprocess: PreprocessI[OO, OS],
    ):
        self.algm = algm
        assert self.algm.name, "agent必须有一个名称(name)"
        self.preprocess = preprocess
        self.name: str = algm.name
        self.algm.on_agent_init({})
        self.reset()

    def reset(self):
        self.preprocess.on_agent_reset()
        self.algm.on_agent_reset()

    def toggleEval(self, newEval: bool):
        self.algm.on_toggle_eval(newEval)

    def set_algm_reporter(self, reporter: Callable[[Dict[str, Any]], None]):
        self.report = reporter
        self.algm.set_reporter(reporter)

    def get_current_state(self, obs_episode: List[OO]) -> OS:
        state = self.preprocess.get_current_state(obs_episode)
        assert isinstance(state, torch.Tensor) or isinstance(
            state,
            LazyFrames), "preprocess.get_current_state应该返回tensor或lazyframes"
        return state

    def format_action(self, a: Union[Action, ActionInfo]) -> ActionInfo:
        if isinstance(a, tuple):
            assert "end" not in a[1]
            assert isinstance(a[0], torch.Tensor)
            a[1]["end"] = False
            return (a[0].detach(), a[1])

        assert isinstance(a, torch.Tensor)
        return (a.detach(), dict(end=False))

    def get_action(self, state: OS, env: gym.Env, mode: Mode, info: Info) -> ActionInfo:
        actinfo = self.format_action(self.algm.take_action(mode, state, info))
        act = actinfo[0]

        action_space = env.action_space

        assert isinstance(action_space, gym.spaces.Discrete) or isinstance(
            action_space, gym.spaces.Box), "目前只能处理Discrete和Box两种action_space"

        if isinstance(action_space, gym.spaces.Discrete):
            assert act.shape == (1, )

        if isinstance(action_space, gym.spaces.Box):
            assert act.shape == action_space.shape

        return actinfo

    def reset_env(self, env: gym.Env, mode: Mode) -> OO:
        o = env.reset()

        if not isinstance(o, tuple):
            self.algm.on_env_reset(mode, dict(env=env, mode=mode))
            return o

        assert len(o) == 2
        assert isinstance(o[1], dict)

        assert 'env' not in o[1]
        o[1]['env'] = env
        assert 'mode' not in o[1]
        o[1]['mode'] = mode

        self.algm.on_env_reset(mode, o[1])
        return o[0]

    def train(self, info: Info) -> int:
        self.toggleEval(False)
        return self.algm.manual_train(info)

    def eval(
        self, env: gym.Env
    ) -> Tuple[ReportInfo, Tuple[List[OO], List[OS], List[Action], List[R],
                                 List[Info]]]:
        self.toggleEval(True)

        observation_episode: List[OO] = []
        state_episode: List[OS] = []
        action_episode: List[Action] = []
        reward_episode: List[R] = []
        info_episode: List[Info] = []

        o = self.reset_env(env, 'eval')
        observation_episode.append(o)
        state_episode.append(self.get_current_state(observation_episode))

        stop = False

        while not stop:

            actinfo = self.get_action(state_episode[-1], env, 'eval', dict(obs=observation_episode, states=state_episode, actions=action_episode, rewards=reward_episode, infos=info_episode))

            act, info = actinfo
            (obs, rwd, stop,
             env_info) = env.step(act[0].cpu().numpy() if isinstance(
                 env.action_space, gym.spaces.Discrete) else act.cpu().numpy())

            info["env_info"] = env_info

            action_episode.append(act)
            info_episode.append(info)
            reward_episode.append(rwd)

            obs = cast(OO, obs)
            observation_episode.append(obs)
            state_episode.append(self.get_current_state(observation_episode))

            assert len(state_episode) == len(observation_episode)

            self.algm.after_step('eval', (
                NotNoneStep(
                    state_episode[-2],
                    action_episode[-1],
                    reward_episode[-1],
                    info_episode[-1],
                ),
                Step(state_episode[-1], None, None, dict(end=stop)),
            ))

        info_episode.append(dict(end=True))
        assert (len(state_episode) == len(observation_episode) ==
                len(action_episode) + 1 == len(reward_episode) + 1 ==
                len(info_episode))

        report_info = self.algm.on_episode_termination('eval', (
            state_episode,
            action_episode,
            reward_episode,
            info_episode,
        )) or dict()

        total_rwd = np.sum([r for r in reward_episode])

        report_info['return'] = total_rwd

        env.close()
        return report_info, (
            observation_episode,
            state_episode,
            action_episode,
            reward_episode,
            info_episode,
        )


AAO = TypeVar("AAO")
AAS = TypeVar("AAS", bound=Union[torch.Tensor, LazyFrames])
AllAgent = Union[Agent[AAO, AAS], OfflineAgent[AAO, AAS]]
