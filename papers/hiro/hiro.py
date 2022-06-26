import setup
import gym
from utils.algorithm import ActionInfo, Mode
from utils.common import Info, Reward, Action
from utils.episode import Episodes
from utils.replay_buffer import ReplayBuffer
from utils.step import NotNoneStep, Step
from utils.transition import (
    Transition,
    TransitionTuple,
)
from torch import nn
import torch
from utils.preprocess import PreprocessI
from utils.algorithm import Algorithm
from typing import Union
from utils.nets import NeuralNetworks, layer_init

from typing import List, Tuple, Any, Optional, Callable, Dict
from papers.sac import NewSAC
import numpy as np
from low_network import LowNetwork
from high_network import HighNetwork

MAX_TIMESTEPS = 500

Observation = torch.Tensor

State = Observation
Goal = torch.Tensor
SUBGOAL_DIM = 15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Preprocess(PreprocessI[Observation, State]):

    def __init__(self):
        pass

    def on_agent_reset(self):
        pass

    def get_current_state(self, h: List[Observation]) -> State:
        assert len(h) > 0

        # assert h[-1].shape == (4, 1, 84, 84)
        return torch.from_numpy(h[-1]).type(torch.float32).to(DEVICE)


class Subgoal:

    def __init__(self, dim):
        limits = np.array([
            -10, -10, -0.5, -1, -1, -1, -1, -0.5, -0.3, -0.5, -0.3, -0.5, -0.3,
            -0.5, -0.3
        ])
        self.shape = (dim, 1)
        self.low = limits[:dim]
        self.high = -self.low
        self.action_dim = dim

    def sample(self):
        return (self.high - self.low) * np.random.sample(
            self.high.shape) + self.low


class Hiro(Algorithm):

    def __init__(self, state_dim: int, goal_dim: int, action_dim: int):
        self.set_name('hiro')
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.subgoal_dim = SUBGOAL_DIM

        self.subgoal = Subgoal(self.subgoal_dim)
        high_action_scale = self.subgoal.high * np.ones(self.subgoal_dim)

        self.high_network = HighNetwork(self.state_dim, self.goal_dim,
                                        self.subgoal_dim, high_action_scale)

        self.low_network = LowNetwork(self.state_dim, self.subgoal_dim,
                                      self.action_dim, 30)

        self.start_training_steps = 2500
        self.buffer_freq = 10
        self.reward_scale = 0.1
        self.train_freq = 10
        self.reset()

    def on_toggle_eval(self, isEval: bool):
        self.eval = isEval
        self.high_network.on_toggle_eval(isEval)
        self.low_network.on_toggle_eval(isEval)

    def reset(self):
        self.replay_buffer_low = ReplayBuffer[Transition](int(2e5))
        self.replay_buffer_high = ReplayBuffer[Transition](int(2e5))

        self.total_steps = 0
        self.train_steps = 0
        self.inner_steps = 0
        self.epoch = 0

        self.reset_episode_info()

    def on_env_reset(self, info: Dict[str, Any]):
        assert self.env is None
        self.env = info['env']
        assert self.fg is None
        self.fg = (torch.from_numpy(info["desired_goal"]).type(
            torch.float32).to(DEVICE))

    def reset_episode_info(self):
        self.env = None
        self.sg = torch.from_numpy(self.subgoal.sample()).type(
            torch.float32).to(DEVICE)
        self.n_sg = None
        self.fg = None
        self.episode_subreward = 0.0
        self.sr = 0
        self.buf = [None, None, None, 0, None, None, [], []]

    def take_action(self, state: State) -> Action:
        s = state.to(DEVICE)
        if self.eval:
            assert self.low_network.eval
            assert self.sg is not None
            return self.low_network.take_action(s, self.sg)

        assert self.env is not None
        a = None
        if self.train_steps <= self.start_training_steps:
            a = torch.from_numpy(self.env.action_space.sample()).type(
                torch.float32).to(DEVICE)
        else:
            assert not self.low_network.eval
            assert self.sg is not None
            a = self.low_network.take_action(s, self.sg)
        assert a is not None

        return a

    def after_step(self, mode: Mode, transition: TransitionTuple[State]):
        (s1, s2) = transition
        self.choose_subgoal(transition)

        assert self.sg is not None
        assert self.n_sg is not None
        self.sr = self.low_reward(s1.state, self.sg, s2.state)
        self.episode_subreward += self.sr

        if mode == 'train':

            self.replay_buffer_low.append(
                Transition((NotNoneStep(
                    s1.state, s1.action, self.sr,
                    dict(goal=self.sg, next_goal=self.n_sg, end=False)),
                            Step(s2.state, None, None,
                                 dict(end=s2.is_end())))))

            if self.inner_steps != 0 and self.inner_steps % self.buffer_freq == 1:
                if len(self.buf[6]) == self.buffer_freq:
                    self.buf[4] = s1.state
                    self.buf[5] = s2.is_end()
                    self.replay_buffer_high.append(
                        Transition((NotNoneStep(
                            self.buf[0], self.buf[2], self.buf[3],
                            dict(goal=self.buf[1],
                                 state_arr=torch.stack(self.buf[6]),
                                 action_arr=torch.stack(self.buf[7]),
                                 end=False)),
                                    Step(self.buf[4], None, None,
                                         dict(end=self.buf[5])))))

                self.buf = [s1.state, self.fg, self.sg, 0, None, None, [], []]

            self.buf[3] += self.reward_scale * s1.reward
            self.buf[6].append(s1.state)
            self.buf[7].append(s1.action)

            if self.train_steps > self.start_training_steps:
                self.train()

            self.train_steps += 1

        assert self.n_sg is not None
        self.sg = self.n_sg
        self.n_sg = None

        self.inner_steps += 1
        self.total_steps += 1

    def train(self):

        self.low_network.train(self.replay_buffer_low)
        if self.train_steps % self.train_freq == 0:
            self.high_network.train(self.replay_buffer_high, self.low_network)

    def low_reward(self, s: torch.Tensor, sg: torch.Tensor,
                   n_s: torch.Tensor) -> float:
        abs_s = s[:sg.shape[0]] + sg
        return -torch.sqrt(torch.sum((abs_s - n_s[:sg.shape[0]])**2)).item()

    def choose_subgoal(self, trx: TransitionTuple[State]):
        assert self.n_sg is None
        (s1, s2) = trx
        if self.eval:
            assert self.high_network.eval
            self.n_sg = self._choose_subgoal(s1.state, self.sg, s2.state)
            return

        assert not self.high_network.eval
        if self.train_steps <= self.start_training_steps:
            self.n_sg = torch.from_numpy(self.subgoal.sample()).type(
                torch.float32).to(DEVICE)
            return

        self.n_sg = self._choose_subgoal(s1.state, self.sg, s2.state)
        return

    def _choose_subgoal(self, s: torch.Tensor, sg: torch.Tensor,
                        n_s: torch.Tensor):
        new_sg = None
        assert self.fg is not None

        if self.train_steps % self.buffer_freq == 0:
            new_sg = self.high_network.take_action(s, self.fg)
        else:
            new_sg = s[:sg.shape[0]] + sg - n_s[:sg.shape[0]]

        return new_sg

    def set_reporter(self, reporter: Callable[[Dict[str, Any]], None]):
        self.reporter = reporter
        self.high_network.set_reporter(reporter)
        self.low_network.set_reporter(reporter)

    def on_episode_termination(self, mode: Mode,
                               sari: Tuple[List[State], List[Action],
                                           List[Reward], List[Info]]):

        (s, _, _, i) = sari
        assert len(s) == MAX_TIMESTEPS + 1
        self.report({mode + '_' + 'episode_subreward': self.episode_subreward})
        self.reset_episode_info()

        self.epoch += 1
        self.inner_steps = 0
