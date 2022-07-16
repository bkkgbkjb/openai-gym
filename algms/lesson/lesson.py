import setup
from utils.algorithm import ActionInfo, Mode
from utils.common import Info, Reward, Action
from utils.episode import Episode
from utils.replay_buffer import ReplayBuffer
from utils.step import NotNoneStep, Step
from utils.transition import (
    TransitionTuple, )
from torch import nn
import torch
from utils.preprocess import PreprocessI
from utils.algorithm import Algorithm
from typing import Literal, Union
from utils.nets import NeuralNetworks, layer_init

from typing import List, Tuple, Any, Optional, Callable, Dict
import numpy as np
from low_network import LowNetwork
from high_network import HighNetwork

MAX_TIMESTEPS = 500
ACTION_SCALE = 1.0

Observation = torch.Tensor

State = Observation
Goal = torch.Tensor

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


class RepresentationNetwork(NeuralNetworks):

    def __init__(self, state_dim: int, goal_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim

        self.net = nn.Sequential(
            layer_init(nn.Linear(self.state_dim, 100)),
            nn.ReLU(),
            layer_init(nn.Linear(100, 100)),
            nn.ReLU(),
            layer_init(nn.Linear(100, self.goal_dim)),
        ).to(DEVICE)

    def forward(self, s: State):
        return self.net(s)


class LESSON(Algorithm):

    def __init__(self, state_dim: int, goal_dim: int, action_dim: int):
        self.set_name('lesson')
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.goal_dim = goal_dim

        self.gamma = 0.99

        self.mini_batch_size = 128

        self.low_network = LowNetwork(self.state_dim, self.goal_dim,
                                      self.action_dim)

        self.high_network = HighNetwork(self.state_dim, self.goal_dim,
                                        self.action_dim)

        self.representation_network = RepresentationNetwork(
            self.state_dim, self.goal_dim)

        self.representation_optim = torch.optim.Adam(
            self.representation_network.parameters(), 1e-4)

        self.c = 50
        self.start_update_phi = 10
        self.reset()

    def on_toggle_eval(self, isEval: bool):
        self.eval = isEval
        self.high_network.on_toggle_eval(isEval)
        self.low_network.on_toggle_eval(isEval)

    def reset(self):
        self.total_steps = 0
        self.inner_steps = 0
        self.epoch = 0
        self.desired_goal = None
        self.representation_goal = None

        self.low_buffer = ReplayBuffer[Episode](2000)

        self.reset_episode_info()

    def reset_episode_info(self):
        self.desired_goal = None
        self.representation_goal = None

        self.has_achieved_goal = False
        self.current_low_input = None
        self.last_high_obs = None
        self.last_high_act = None
        self.high_reward = 0.0

        self.env_id = None

    def on_env_reset(self, mode: Mode, info: Dict[str, Any]):
        assert self.desired_goal is None
        self.desired_goal = (torch.from_numpy(info["desired_goal"]).type(
            torch.float32).to(DEVICE))

        obs = torch.from_numpy(info["observation"]).type(
            torch.float32).to(DEVICE)
        
        assert self.representation_goal is None
        self.representation_goal = (self.representation_network(
            obs.unsqueeze(0)).squeeze(0)).detach()
        
        assert self.env_id is None
        self.env_id: Union[Literal['eval_random'], Literal['train'], Literal['eval_farthest']] = info['env']._env_id


    def take_action(self, mode: Mode, state: State) -> ActionInfo:
        s = state.to(DEVICE)
        assert self.desired_goal is not None
        if self.inner_steps % self.c == 0:
            if self.last_high_act is not None:
                assert self.last_high_obs is not None

                self.high_network.after_step(mode, (
                    NotNoneStep(self.last_high_obs, self.last_high_act,
                                self.high_reward),
                    Step(torch.cat([s, self.desired_goal]), None, None,
                         dict(end=self.has_achieved_goal)),
                ))
                if mode == 'train' and self.high_network.sac.replay_memory.len >= 128:
                    self.high_network.manual_train(dict())

            self.high_reward = 0.0
            self.has_achieved_goal = False

            assert self.eval == self.high_network.eval
            act = self.high_network.take_action(mode, s, self.desired_goal)

            self.current_low_input = (self.representation_goal + act).clip(-200, 200)
            self.last_high_obs = torch.cat([
                state,
                self.desired_goal,
            ])
            self.last_high_act = act

        with torch.no_grad():
            assert self.current_low_input is not None
            assert self.eval == self.low_network.eval
            act = self.low_network.take_action(
                mode,
                s.unsqueeze(0),
                self.current_low_input.unsqueeze(0).to(DEVICE)
            )

            return act, dict(rg=self.representation_goal,
                             low_input=self.current_low_input)

    def after_step(self, mode: Mode, transition: TransitionTuple[State]):
        (s1, s2) = transition

        if s1.info["env_info"]["is_success"]:
            self.has_achieved_goal = True

        if not self.has_achieved_goal:
            self.high_reward += s1.reward

        self.representation_goal = (self.representation_network(
            s2.state.unsqueeze(0)).squeeze(0).to(DEVICE)).detach()

        if mode == 'train' and self.epoch > self.start_update_phi:
            self.update_phi()

        self.inner_steps += 1
        self.total_steps += 1

    def on_episode_termination(self, mode: Mode, sari: Tuple[List[State], List[Action],
                                                 List[Reward], List[Info]]):

        (s, _, _, i) = sari
        assert len(s) == MAX_TIMESTEPS + 1
        assert self.last_high_obs is not None
        assert self.last_high_act is not None

        assert 'rg' not in i[-1]
        i[-1]['rg'] = self.representation_goal

        assert self.desired_goal is not None

        self.high_network.after_step(mode, (
            NotNoneStep(self.last_high_obs, self.last_high_act,
                        self.high_reward),
            Step(
                torch.cat([s[-1], self.desired_goal]),
                None,
                None,
                dict(end=self.has_achieved_goal),
            ),
        ))

        self.high_network.on_episode_termination(mode, sari)

        if mode == 'train':
            self.low_buffer.append(self.get_episodes(sari))
            for _ in range(200):
                self.low_network.manual_train(dict(buffer=self.low_buffer))

        self.reset_episode_info()

        self.epoch += 1
        self.inner_steps = 0
    
    def manual_train(self, info: Dict[str, Any]):
        pass

    def collect_samples(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = 100
        episodes = self.low_buffer.sample(batch_size)

        ts = np.random.randint(MAX_TIMESTEPS - self.c, size=batch_size)

        hi_obs = obs = torch.stack(
            [e.steps[ts[i]].state for i, e in enumerate(episodes)])

        hi_obs_next = torch.stack(
            [e.steps[ts[i] + self.c].state for i, e in enumerate(episodes)])
        obs_next = torch.stack(
            [e.steps[ts[i] + 1].state for i, e in enumerate(episodes)])

        assert obs.shape == obs_next.shape == hi_obs.shape == hi_obs_next.shape == (batch_size, self.state_dim)
        return (obs, obs_next, hi_obs, hi_obs_next)

    def update_phi(self):
        (obs, obs_next, hi_obs, hi_obs_next) = self.collect_samples()

        new_obs = self.representation_network(obs)
        new_obs_next = self.representation_network(obs_next)
        min_dist = (new_obs - new_obs_next).pow(2).mean(dim=1).clamp(min=0)

        new_hi_obs = self.representation_network(hi_obs)
        new_hi_obs_next = self.representation_network(hi_obs_next)

        max_dist = (1 -
                    (new_hi_obs - new_hi_obs_next).pow(2).mean(dim=1)).clamp(
                        min=0)

        representation_loss = (min_dist + max_dist).mean()

        self.representation_optim.zero_grad()
        representation_loss.backward()
        self.representation_optim.step()

        self.report(dict(representation_loss=representation_loss))

    def set_reporter(self, reporter: Callable[[Dict[str, Any]], None]):
        self.reporter = reporter
        self.high_network.set_reporter(reporter)
        self.low_network.set_reporter(reporter)

    def get_episodes(
        self, sari: Tuple[List[State], List[Action], List[Reward], List[Info]]
    ) -> Episode[State]:
        # episode = Episodes[State]()
        episode = Episode[State].from_list(sari)

        for i in range(episode.len - 1):
            episode.steps[i].info['next_obs'] = episode.steps[i +
                                                                    1].state
            episode.steps[i].info['next_rg'] = episode.steps[
                i + 1].info['rg']

        return episode

