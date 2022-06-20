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

MAX_TIMESTEPS = 500

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
        assert a.size(1) == self.action_dim
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
        assert s.size(0) == g.size(0)
        x = torch.cat([s, g], dim=1)
        act = self.net(x)
        assert act.shape == (s.size(0), self.action_dim)
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

    def pertub(self, act: np.ndarray):
        act += 0.2 * 1.0 * np.random.randn(self.action_dim)
        return act.clip(-1.0, 1.0)

    def sample(self, buffers: ReplayBuffer[Episodes[State]]):
        episodes = buffers.sample(128)

        time_stamps = np.random.randint(MAX_TIMESTEPS - 1, size=128)

        sampled_steps = [
            e.get_step(time_stamps[i]) for i, e in enumerate(episodes)
        ]

        obs = torch.stack([s.state for s in sampled_steps])

        obs_next = torch.stack([s.info['next_obs'] for s in sampled_steps])

        acts = torch.stack([
            torch.from_numpy(s.action).type(torch.float32).to(DEVICE)
            for s in sampled_steps
        ])

        rgs = torch.stack(
            [s.info['representation_goal'] for s in sampled_steps])

        rg_next = torch.stack([s.info['next_rg'] for s in sampled_steps])

        g = torch.stack([
            torch.from_numpy(s.info['high_act']).type(torch.float32).to(DEVICE)
            for s in sampled_steps
        ])

        reward = torch.from_numpy(
            -np.linalg.norm(rg_next.cpu().numpy() - g.cpu().numpy(), axis=-1) *
            0.1).type(torch.float32).to(DEVICE)

        g_next = g
        not_done = (torch.norm(
            (rg_next - g_next), dim=1) > 0.1).type(torch.int8).reshape(-1, 1)
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

        self.report(dict(actor_loss=actor_loss, critic_loss=critic_loss))


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
        return ((rg.cpu().numpy() + act).clip(-200, 200), dict(raw_action=act))

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
            obs.unsqueeze(0)).squeeze(0))

        self.reset_episode_info()

    def reset_episode_info(self):
        self.has_achieved_goal = False
        self.current_high_act = None
        self.last_high_obs = None
        self.last_high_act = None
        self.high_reward = 0.0

    @torch.no_grad()
    def take_action(self, state: State) -> ActionInfo:
        s = state.to(DEVICE)
        if self.inner_steps % self.c == 0:
            (act,
             info) = self.high_network.take_action(s, self.desired_goal,
                                                   self.representation_goal)

            self.current_high_act = act
            self.last_high_obs = torch.cat([
                state,
                self.desired_goal,
            ])
            self.last_high_act = info["raw_action"]

        assert self.current_high_act is not None
        act = self.low_network.take_action(
            s.unsqueeze(0),
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

        max_dist = (1 - (new_hi_obs - new_hi_obs_next)).pow(2).mean(
            dim=1).clamp(min=0)

        representation_loss = (min_dist + max_dist).mean()

        self.representation_optim.zero_grad()
        representation_loss.backward()
        self.representation_optim.step()

        self.report(dict(representation_loss=representation_loss))

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
                i + 1).info['representation_goal']

        return episode

    def on_episode_termination(self, sari: Tuple[List[State], List[Action],
                                                 List[Reward], List[Info]]):

        (s, _, _, _) = sari
        assert len(s) == MAX_TIMESTEPS + 1
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

        self.low_buffer.append(self.get_episodes(sari))
        for _ in range(20):
            self.low_network.train(self.low_buffer)

        self.reset_episode_info()

        self.epoch += 1
        self.inner_steps = 0
