import numpy as np
import torch
import gym
from utils.env_sb3 import LazyFrames, resolve_lazy_frames
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
from utils.common import ActionInfo, AllowedState, Step, EpisodeGeneric, Action, Reward

O = TypeVar("O")
S = TypeVar("S", bound=AllowedState)
R = Reward


class Agent(Generic[O, S]):

    def __init__(
        self,
        env: gym.Env,
        algm: AlgorithmInterface[S],
        preprocess: PreprocessInterface[O, S],
    ):
        self.env = env
        self.algm = algm
        self.preprocess = preprocess
        self.name: str = algm.name
        self.reset()

    def get_current_state(self, obs_episode: List[O]):
        state = self.preprocess.get_current_state(obs_episode)
        assert isinstance(state, torch.Tensor) or isinstance(
            state,
            LazyFrames), "preprocess.get_current_state应该返回tensor或lazyframes"
        return state

    def reset(self, ):
        self.ready_act: Optional[ActionInfo] = None
        self.end = False

        self.observation_episode: List[O] = []
        self.state_episode: List[S] = []
        self.action_episode: List[ActionInfo] = []
        self.reward_episode: List[R] = []

        self.eval_observation_episode: List[O] = []

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

    def eval(self, env: gym.Env) -> Tuple[float, Tuple[List[O]]]:
        assert len(self.eval_observation_episode) == 0
        assert not self.end, "should reset before eval agnet"
        self.toggleEval(True)

        o = cast(O, env.reset())
        self.eval_observation_episode.append(o)

        s = False
        rwd = 0.0

        while not s:
            actinfo = self.get_action(
                self.get_current_state(self.eval_observation_episode))
            act = actinfo[0]

            (o, r, s, _) = env.step(
                act[0] if isinstance(env.action_space, gym.spaces.Discrete) else act)
            rwd += r
            self.eval_observation_episode.append(o)

        self.report({'eval_return': rwd})
        return rwd, (self.eval_observation_episode, )

    def get_action(self, state: S) -> ActionInfo:
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
    ) -> Tuple[float, Tuple[List[O], List[S], List[ActionInfo], List[R]]]:
        assert not self.end, "agent needs to be reset before training"
        self.toggleEval(False)

        o = cast(O, self.env.reset())
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

            obs = cast(O, obs)
            self.observation_episode.append(obs)
            self.state_episode.append(
                self.get_current_state(self.observation_episode))

            self.ready_act = (None if stop else self.get_action(
                self.state_episode[-1]))

            assert len(self.state_episode) == len(self.observation_episode)

            self.algm.after_step(
                (
                    self.state_episode[-2],
                    self.action_episode[-1],
                    self.reward_episode[-1],
                ),
                (
                    self.state_episode[-1],
                    self.ready_act,
                ),
            )

        assert len(self.state_episode) == len(self.observation_episode) == len(
            self.action_episode) + 1 == len(self.reward_episode) + 1

        self.end = True

        self.algm.on_episode_termination(
            (self.state_episode, self.action_episode, self.reward_episode))

        total_rwd = np.sum([r for r in self.reward_episode])

        self.report({"train_return": total_rwd})

        return total_rwd, (self.observation_episode, self.state_episode,
                           self.action_episode, self.reward_episode)

    def render(self, mode: str):
        self.env.render(mode)

    def close(self):
        self.reset()
        self.env.close()
