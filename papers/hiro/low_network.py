import setup
from utils.episode import Episodes
from utils.replay_buffer import ReplayBuffer
from torch import nn
import torch
from utils.algorithm import Algorithm
from utils.nets import NeuralNetworks, layer_init
import torch.nn.functional as F
import numpy as np

from utils.transition import Transition, resolve_transitions

MAX_TIMESTEPS = 500

Observation = torch.Tensor
Action = np.ndarray

State = Observation
Goal = torch.Tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LowActor(NeuralNetworks):

    def __init__(self, state_dim, goal_dim, action_dim, scale):
        super(LowActor, self).__init__()
        self.scale = scale
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim

        self.l1 = layer_init(nn.Linear(state_dim + goal_dim, 300)).to(DEVICE)
        self.l2 = layer_init(nn.Linear(300, 300)).to(DEVICE)
        self.l3 = layer_init(nn.Linear(300, action_dim)).to(DEVICE)

    def forward(self, state, goal):
        a = F.relu(self.l1(torch.cat([state.to(DEVICE), goal.to(DEVICE)], 1)))
        a = F.relu(self.l2(a))
        return self.scale * torch.tanh(self.l3(a))


class LowCritic(NeuralNetworks):

    def __init__(self, state_dim, goal_dim, action_dim):
        super(LowCritic, self).__init__()

        self.l1 = layer_init(nn.Linear(state_dim + goal_dim + action_dim,
                                       300)).to(DEVICE)
        self.l2 = layer_init(nn.Linear(300, 300)).to(DEVICE)
        self.l3 = layer_init(nn.Linear(300, 1)).to(DEVICE)

    def forward(self, state, goal, action):
        sa = torch.cat([state.to(DEVICE),
                        goal.to(DEVICE),
                        action.to(DEVICE)], 1)

        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)

        return q


class LowNetwork(Algorithm):

    def __init__(self, state_dim: int, goal_dim: int, action_dim: int,
                 action_scale: float):
        self.set_name('low-network')
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.action_scale = action_scale

        self.expl_noise = 0.333
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.gamma = 0.99
        self.policy_freq = 2
        self.tau = 5e-3
        self.batch_size = 128

        self.actor = LowActor(self.state_dim, self.goal_dim, self.action_dim,
                              self.action_scale)
        self.actor_target = self.actor.clone().no_grad()

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic1 = LowCritic(self.state_dim, self.goal_dim,
                                 self.action_dim)
        self.critic1_target = self.critic1.clone().no_grad()
        self.critic1_loss = nn.SmoothL1Loss()
        self.critic1_optim = torch.optim.Adam(
            self.critic1.parameters(),
            lr=1e-3,
        )

        self.critic2 = LowCritic(self.state_dim, self.goal_dim,
                                 self.action_dim)
        self.critic2_target = self.critic2.clone().no_grad()
        self.critic2_loss = nn.SmoothL1Loss()
        self.critic2_optim = torch.optim.Adam(
            self.critic2.parameters(),
            lr=1e-3,
        )

        self.train_times = 0
        self.eval = False

    def on_toggle_eval(self, isEval: bool):
        self.eval = isEval

    def take_action(self, s: torch.Tensor, g: torch.Tensor):
        s = s.unsqueeze(0)
        g = g.unsqueeze(0)
        if self.eval:
            return self.actor(s, g).squeeze()

        act = self.actor(s, g)
        act += self.pertub(act)

        return act.clamp(-self.action_scale, self.action_scale).squeeze()

    def policy(self, s: torch.Tensor, g: torch.Tensor):
        return self.actor(s, g).squeeze()

    def pertub(self, act: torch.Tensor):
        mean = torch.zeros_like(act)
        var = torch.ones_like(act)
        return self.action_scale * torch.normal(
            mean, self.expl_noise * var).to(DEVICE)

    def train(self, buffer: ReplayBuffer[Transition]):
        (states, actions, rewards, n_states, done,
         infos) = resolve_transitions(buffer.sample(self.batch_size),
                                      (self.state_dim, ), (self.action_dim, ))

        goals = torch.stack([i['goal'] for i in infos]).detach()
        n_goals = torch.stack([i['next_goal'] for i in infos]).detach()
        not_done = 1 - done

        with torch.no_grad():
            noise = (self.action_scale * torch.randn_like(actions) *
                     self.policy_noise).clamp(
                         -self.noise_clip * self.action_scale,
                         self.noise_clip * self.action_scale)

            n_actions = (self.actor_target(n_states, n_goals) + noise).clamp(
                -self.action_scale, self.action_scale)

            target_Q1 = self.critic1_target(n_states, n_goals, n_actions)
            target_Q2 = self.critic2_target(n_states, n_goals, n_actions)

            target_Q = torch.min(target_Q1, target_Q2)

            target_val = (rewards + not_done * self.gamma * target_Q).detach()

        current_Q1 = self.critic1(states, goals, actions)
        current_Q2 = self.critic2(states, goals, actions)

        critic1_loss = self.critic1_loss(current_Q1, target_val)
        critic2_loss = self.critic2_loss(current_Q2, target_val)

        critic_loss = critic1_loss + critic2_loss

        self.critic1_optim.zero_grad()
        self.critic2_optim.zero_grad()
        critic_loss.backward()
        self.critic1_optim.step()
        self.critic2_optim.step()

        self.report(dict(critic_loss=critic_loss))

        if self.train_times % self.policy_freq == 0:
            a = self.actor(states, goals)
            Q1 = self.critic1(states, goals, a)

            actor_loss = -Q1.mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            self.critic1_target.soft_update_to(self.critic1, self.tau)
            self.critic2_target.soft_update_to(self.critic2, self.tau)

            self.actor_target.soft_update_to(self.actor, self.tau)
            self.report(dict(actor_loss=actor_loss))

        self.train_times += 1
