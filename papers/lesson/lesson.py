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


class LowCritic(NeuralNetworks):

    def __init__(self, state_dim: int, goal_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim + goal_dim + action_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1)),
        ).to(DEVICE)

        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim

    def forward(self, s: State, g: Goal, a: torch.Tensor) -> torch.Tensor:
        assert s.size(1) == self.state_dim
        assert a.shape[0] == self.action_dim
        assert g.size(1) == self.goal_dim
        return self.net(
            torch.cat([s.to(DEVICE), g.to(DEVICE),
                       a.to(DEVICE)], 1))


class LowActor(NeuralNetworks):

    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
    ):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim + goal_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, action_dim)),
            nn.Tanh(),
        ).to(DEVICE)

        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim

    def forward(self, s: State, g: torch.Tensor) -> torch.Tensor:
        assert s.size(1) == self.state_dim
        assert g.size(1) == self.goal_dim
        x = torch.cat([s, g], dim=1)
        act = self.net(x)
        assert act.shape == (1, self.action_dim)
        return act


class RepresentationNetwork(NeuralNetworks):

    def __init__(self, state_dim: int, goal_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim

        self.net = nn.Sequential(
            nn.Linear(self.state_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, self.goal_dim),
        ).to(DEVICE)

    def forward(self, s: State):
        return self.net(s)


class LowNetwork(Algorithm):

    def __init__(self, state_dim: int, goal_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim

        self.actor = LowActor(self.state_dim, self.goal_dim, self.action_dim)
        self.actor_target = self.actor.clone().no_grad()

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=2e-4)

        self.critic = LowCritic(self.state_dim, self.goal_dim, self.action_dim)
        self.critic_target = self.critic.clone().no_grad()

        self.critic_loss = nn.MSELoss()

        self.critic_optim = torch.optim.Adam(self.critic.parameters(),
                                             lr=2e-4,
                                             weight_decay=1e-5)

        self.training = True
        self.eps = 0.2
        self.gamma = 0.99

    @torch.no_grad()
    def take_action(self, s: torch.Tensor, g: torch.Tensor):
        if self.training and np.random.rand() < self.eps:
            return np.random.uniform(-1, 1, self.action_dim)

        act = self.actor(s, g).cpu().detach().squeeze(0).numpy()
        if self.training:
            return self.pertub(act)

        return act

    def after_step(self, transition: TransitionTuple[State]):
        pass

    def pertub(self, act: np.ndarray):
        act += 0.2 * 1.0 * np.random.randn(self.action_dim)
        return act.clip(-1.0, 1.0)

    def sample(self, buffers: ReplayBuffer[Episodes[State]]):
        episodes = buffers.sample(128)
        time_stamps = np.random.randint(3000, size=128)

        sampled_episodes = [
            e.get_steps(time_stamps.tolist()) for e in episodes
        ]

        # obs = [s. for s in sampled_steps]
        obs = torch.as_tensor([[s.state for s in e] for e in sampled_episodes])

        obs_next = obs[1:]
        acts = torch.as_tensor([(s.action for s in e)
                                for e in sampled_episodes])

        rgs = torch.as_tensor([(s.info['representation_goal'] for s in e)
                               for e in sampled_episodes])

        rg_next = rgs[1:]

        g = torch.as_tensor([(s.info['hi_act'] for s in e)
                             for e in sampled_episodes])

        reward = torch.from_numpy(
            -np.linalg.norm(rg_next.numpy() - g.numpy(), axis=-1) * 0.1).type(
                torch.float32)

        g_next = g
        not_done = (torch.norm(
            (rg_next - g_next), dim=1) > 0.1).astype(torch.int8).reshape(
                -1, 1)
        return (obs, obs_next, rgs, rg_next, acts, reward, g, g_next, not_done)

    def train(self, low_buffer: ReplayBuffer[Episodes]):
        (obs, obs_next, rgs, rg_next, acts, reward, g, g_next,
         not_done) = self.sample(low_buffer)

        with torch.no_grad():
            actions_next = self.actor_target(obs_next, g_next)
            q_next_value = self.critic_target(obs_next, g_next,
                                              actions_next).detach()

            target_q_value = (reward +
                              self.gamma * q_next_value * not_done).detach()

        real_q_value = self.critic(obs, g, acts)

        critic_loss = (target_q_value - real_q_value).pow(2).mean()

        actions_real = self.actor(obs, g)
        actor_loss = -self.critic(obs, g, actions_real).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optim.step()


class HighNetwork(Algorithm):

    def __init__(self, state_dim: int, goal_dim: int, action_dim: int) -> None:
        self.name = "lesson-high"

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.goal_dim = goal_dim

        self.sac = NewSAC(self.state_dim + self.goal_dim, self.goal_dim, 1.0,
                          True)

        self.random_episode = 20
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
        return ((rg.numpy() + act).clip(-200, 200), dict(raw_action=act))

    def after_step(self, transition: TransitionTuple[State]):
        self.sac.after_step(transition)
        if self.times >= 128:
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

        self.tau = 5e-3

        self.start_traininig_size = int(1e4)
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
        self.high_random_episode = 20
        self.start_update_phi = 10
        self.reset()

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
            obs.unsqueeze(0)).squeeze(0).to(DEVICE))

        self.reset_episode_info()

    def reset_episode_info(self):
        self.has_achieved_goal = False
        self.current_high_act = None
        self.last_high_obs = None
        self.last_high_act = None
        self.high_reward = 0.0

    def update_high(self):
        pass

    @torch.no_grad()
    def take_action(self, state: State) -> ActionInfo:
        if self.inner_steps % self.c == 0:
            (act,
             info) = self.high_network.take_action(state.to(DEVICE),
                                                   self.desired_goal,
                                                   self.representation_goal)

            self.current_high_act = act
            self.last_high_obs = torch.cat([
                state,
                self.desired_goal,
            ])
            self.last_high_act = info["raw_action"]

        assert self.current_high_act is not None
        act = self.low_network.take_action(
            state.unsqueeze(0).to(DEVICE),
            torch.from_numpy(self.current_high_act).type(
                torch.float32).unsqueeze(0).to(DEVICE),
        )
        self.representation_goal = (self.representation_network(
            state.unsqueeze(0)).squeeze(0).to(DEVICE))

        return act, dict(representation_goal=self.representation_goal,
                         high_act=self.current_high_act)

    def collect_samples(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        episodes = self.low_buffer.sample(100)
        ts = np.random.randint(3000 - self.c, size=100).tolist()
        hi_obs = obs = [(s.state for s in e.get_steps(ts)) for e in episodes]
        hi_obs_next = [
            (s.state
             for s in e.get_steps(list(map(lambda _ts: _ts + self.c, ts))))
            for e in episodes
        ]
        obs_next = [(s.state
                     for s in e.get_steps(list(map(lambda _ts: _ts + 1, ts))))
                    for e in episodes]

        return (
            torch.tensor(obs, dtype=torch.float32),
            torch.tensor(obs_next, dtype=torch.float32),
            torch.tensor(hi_obs, dtype=torch.float32),
            torch.tensor(hi_obs_next, dtype=torch.float32),
        )

    def update_phi(self):
        (obs, obs_next, hi_obs, hi_obs_next) = self.collect_samples()

        new_obs = self.representation_network(obs)
        new_obs_next = self.representation_network(obs_next)
        min_dist = (new_obs - new_obs_next).pow(2).mean(dim=1).clamp(min=0)

        new_hi_obs = self.representation_network(hi_obs)
        new_hi_obs_next = self.representation_network(hi_obs_next)

        max_dist = (1 - (new_hi_obs - new_hi_obs_next)).pow(2).mean(
            dim=1).clamp(min=0)

        representation_loss = (min_dist + max_dist).mean()

        self.representation_optim.zero_grad()
        representation_loss.backward()
        self.representation_optim.step()

        # self.re

    def after_step(self, transition: TransitionTuple[State]):
        (s1, s2) = transition

        if s1.info["env_info"]["is_success"]:
            self.has_achieved_goal = True

        self.high_reward += s1.reward

        if (self.inner_steps != 0 and self.inner_steps % self.c == self.c - 1
                and not s2.is_end()):
            assert self.last_high_obs is not None
            assert self.last_high_act is not None

            self.high_network.after_step((
                NotNoneStep(self.last_high_obs, self.last_high_act,
                            self.high_reward),
                Step(
                    torch.cat([s2.state, self.desired_goal]),
                    None,
                    None,
                ),
            ))

            self.high_reward = 0.0

        if self.epoch > self.start_update_phi:
            self.update_phi()

        self.inner_steps += 1
        self.total_steps += 1

    def set_reporter(self, reporter: Callable[[Dict[str, Any]], None]):
        self.high_network.set_reporter(reporter)
        self.low_network.set_reporter(reporter)

    def on_episode_termination(self, sari: Tuple[List[State], List[Action],
                                                 List[Reward], List[Info]]):
        (s, a, r, i) = sari
        assert self.last_high_obs is not None
        assert self.last_high_act is not None

        self.high_network.after_step((
            NotNoneStep(self.last_high_obs, self.last_high_act,
                        self.high_reward),
            Step(
                torch.cat([s[-1], self.desired_goal]),
                None,
                None,
                dict(end=True),
            ),
        ))

        self.high_network.on_episode_termination(sari)

        episode = Episodes()
        self.low_buffer.append(episode.from_list(sari))
        for _ in range(20):
            self.low_network.train(self.low_buffer)

        self.reset_episode_info()

        self.epoch += 1
        self.inner_steps = 0
