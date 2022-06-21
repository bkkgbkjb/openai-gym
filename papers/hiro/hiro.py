import setup
from utils.algorithm import ActionInfo
from utils.common import Info, Reward
from utils.episode import Episodes
from utils.replay_buffer import ReplayBuffer
from utils.step import NotNoneStep, Step
from utils.transition import (
    TransitionTuple, )
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

MAX_TIMESTEPS = 500
ACTION_SCALE = 1.0

Observation = torch.Tensor
Action = np.ndarray

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


class HighNetwork(Algorithm):

    def __init__(self, state_dim: int, goal_dim: int, action_dim: int) -> None:
        self.name = "lesson-high"

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.goal_dim = goal_dim

        self.sac = NewSAC(self.state_dim + self.goal_dim, self.goal_dim, 20.0,
                          True, False)

        self.random_episode = 300
        self.reset()

    def set_reporter(self, reporter: Callable[[Dict[str, Any]], None]):
        self.sac.set_reporter(reporter)

    def reset(self):
        self.times = 0
        self.epoch = 0

    @torch.no_grad()
    def take_action(self, s: State, dg: Goal, rg: Goal) -> ActionInfo:

        act = None
        if self.epoch <= self.random_episode:
            act = np.random.uniform(-20, 20, self.goal_dim)

        else:
            obs = torch.cat([s, dg])
            assert obs.shape == (self.state_dim + self.goal_dim, )
            act = self.sac.take_action(obs)

        assert act.shape == (self.goal_dim, )
        return ((rg.cpu().numpy() + act).clip(-200, 200), dict(raw_action=act))

    def after_step(self, transition: TransitionTuple[State]):
        self.sac.after_step(transition)
        if self.sac.replay_memory.len >= 128:
            self.sac.train()

        self.times += 1

    def on_episode_termination(self, sari: Tuple[List[State], List[Action],
                                                 List[Reward], List[Info]]):
        self.epoch += 1


class LESSON(Algorithm):

    def __init__(self, state_dim: int, goal_dim: int, action_dim: int):
        self.name = "lesson"
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

        self.low_buffer = ReplayBuffer[Episodes](300)

        self.reset_episode_info()

    @torch.no_grad()
    def on_env_reset(self, info: Dict[str, Any]):
        self.desired_goal = (torch.from_numpy(info["desired_goal"]).type(
            torch.float32).to(DEVICE))

        obs = torch.from_numpy(info["observation"]).type(
            torch.float32).to(DEVICE)
        self.representation_goal = (self.representation_network(
            obs.unsqueeze(0)).squeeze(0)).detach()

        self.reset_episode_info()

    def reset_episode_info(self):
        self.has_achieved_goal = False
        self.current_high_act = None
        self.last_high_obs = None
        self.last_high_act = None
        self.high_reward = 0.0

    def take_action(self, state: State) -> ActionInfo:
        s = state.to(DEVICE)
        if self.inner_steps % self.c == 0:
            if self.last_high_act is not None:
                assert self.current_high_act is not None
                assert self.last_high_obs is not None

                self.high_network.after_step((
                    NotNoneStep(self.last_high_obs, self.last_high_act,
                                self.high_reward),
                    Step(torch.cat([s, self.desired_goal]), None, None,
                         dict(end=self.has_achieved_goal)),
                ))

            self.high_reward = 0.0

            (act,
             info) = self.high_network.take_action(s, self.desired_goal,
                                                   self.representation_goal)

            self.current_high_act = act
            self.last_high_obs = torch.cat([
                state,
                self.desired_goal,
            ])
            self.last_high_act = info["raw_action"]

        with torch.no_grad():
            assert self.current_high_act is not None
            act = self.low_network.take_action(
                s.unsqueeze(0),
                torch.from_numpy(self.current_high_act).type(
                    torch.float32).unsqueeze(0).to(DEVICE),
            )

            return act, dict(rg=self.representation_goal,
                             high_act=self.current_high_act)

    def collect_samples(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        episodes = self.low_buffer.sample(100)

        ts = np.random.randint(MAX_TIMESTEPS - self.c, size=100)

        hi_obs = obs = torch.stack(
            [e.get_step(ts[i]).state for i, e in enumerate(episodes)])

        hi_obs_next = torch.stack(
            [e.get_step(ts[i] + self.c).state for i, e in enumerate(episodes)])
        obs_next = torch.stack(
            [e.get_step(ts[i] + 1).state for i, e in enumerate(episodes)])

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

    def after_step(self, transition: TransitionTuple[State]):
        (s1, s2) = transition

        if s1.info["env_info"]["is_success"]:
            self.has_achieved_goal = True

        if not self.has_achieved_goal:
            self.high_reward += s1.reward

        self.representation_goal = (self.representation_network(
            s2.state.unsqueeze(0)).squeeze(0).to(DEVICE)).detach()

        if self.epoch > self.start_update_phi:
            self.update_phi()

        self.inner_steps += 1
        self.total_steps += 1

    def set_reporter(self, reporter: Callable[[Dict[str, Any]], None]):
        self.reporter = reporter
        self.high_network.set_reporter(reporter)
        self.low_network.set_reporter(reporter)

    def get_episodes(
        self, sari: Tuple[List[State], List[Action], List[Reward], List[Info]]
    ) -> Episodes[State]:
        episode = Episodes[State]()
        episode.from_list(sari)

        for i in range(episode.len - 1):
            episode.get_step(i).info['next_obs'] = episode.get_step(i +
                                                                    1).state
            episode.get_step(i).info['next_rg'] = episode.get_step(
                i + 1).info['rg']

        return episode

    def on_episode_termination(self, sari: Tuple[List[State], List[Action],
                                                 List[Reward], List[Info]]):

        (s, _, _, i) = sari
        assert len(s) == MAX_TIMESTEPS + 1
        assert self.last_high_obs is not None
        assert self.last_high_act is not None

        assert 'rg' not in i[-1]
        i[-1]['rg'] = self.representation_goal

        self.high_network.after_step((
            NotNoneStep(self.last_high_obs, self.last_high_act,
                        self.high_reward),
            Step(
                torch.cat([s[-1], self.desired_goal]),
                None,
                None,
                dict(end=self.has_achieved_goal),
            ),
        ))

        self.high_network.on_episode_termination(sari)

        self.low_buffer.append(self.get_episodes(sari))
        for _ in range(200):
            self.low_network.train(self.low_buffer)

        self.reset_episode_info()

        self.epoch += 1
        self.inner_steps = 0
