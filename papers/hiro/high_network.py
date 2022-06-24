import setup
from utils.episode import Episodes
from utils.replay_buffer import ReplayBuffer
from torch import nn
import torch
from utils.algorithm import Algorithm
from utils.nets import NeuralNetworks, layer_init
from utils.transition import resolve_transitions
import torch.nn.functional as F
import numpy as np

from low_network import LowNetwork
from utils.transition import Transition

MAX_TIMESTEPS = 500
ACTION_SCALE = 30.0

Observation = torch.Tensor
Action = np.ndarray

State = Observation
Goal = torch.Tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class HighActor(NeuralNetworks):

    def __init__(self, state_dim, goal_dim, action_dim, scale):
        super(HighActor, self).__init__()
        self.scale = scale.to(DEVICE)

        self.l1 = layer_init(nn.Linear(state_dim + goal_dim, 300)).to(DEVICE)
        self.l2 = layer_init(nn.Linear(300, 300)).to(DEVICE)
        self.l3 = layer_init(nn.Linear(300, action_dim)).to(DEVICE)

    def forward(self, state, goal):
        a = F.relu(self.l1(torch.cat([state.to(DEVICE), goal.to(DEVICE)], 1)))
        a = F.relu(self.l2(a))
        return self.scale * torch.tanh(self.l3(a))


class HighCritic(NeuralNetworks):

    def __init__(self, state_dim, goal_dim, action_dim):
        super(HighCritic, self).__init__()

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


class HighNetwork(Algorithm):

    def __init__(self, state_dim: int, goal_dim: int, action_dim: int,
                 action_scale: np.ndarray):
        self.set_name('high-network')
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.action_scale = torch.from_numpy(action_scale).type(
            torch.float32).to(DEVICE)

        self.expl_noise = 0.1
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.gamma = 0.99
        self.policy_freq = 2
        self.tau = 5e-3
        self.batch_size = 128

        self.actor = HighActor(self.state_dim, self.goal_dim, self.action_dim,
                               self.action_scale)
        self.actor_target = self.actor.clone().no_grad()

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic1 = HighCritic(self.state_dim, self.goal_dim,
                                  self.action_dim)
        self.critic1_target = self.critic1.clone().no_grad()
        self.critic1_loss = nn.MSELoss()
        self.critic1_optim = torch.optim.Adam(
            self.critic1.parameters(),
            lr=1e-3,
        )

        self.critic2 = HighCritic(self.state_dim, self.goal_dim,
                                  self.action_dim)
        self.critic2_target = self.critic2.clone().no_grad()
        self.critic2_loss = nn.MSELoss()
        self.critic2_optim = torch.optim.Adam(
            self.critic2.parameters(),
            lr=1e-3,
        )

        self.candidate_goals = 16

        self.train_times = 0
        self.eval = False

    def off_policy_correct(self, low_network: LowNetwork, sgoals: torch.Tensor,
                           states: torch.Tensor,
                           actions: torch.Tensor) -> torch.Tensor:

        action_scale = self.action_scale
        first_s = states[:, 0]
        last_s = states[:, -1]

        assert (last_s - first_s).unsqueeze(1).shape == (self.batch_size, 1,
                                                         self.state_dim)
        diff_goal = (
            last_s -
            first_s).unsqueeze(1)[:, :, :self.action_dim]  #.unsqueeze(
        #1)  #[:, np.newaxis, :self.action_dim]

        original_goal = sgoals.unsqueeze(1)  #[:, np.newaxis, :]
        random_goals = torch.from_numpy(
            np.random.normal(
                loc=diff_goal.cpu().numpy(),
                scale=0.333 * action_scale.unsqueeze(0).unsqueeze(
                    0).cpu().numpy(),  #[np.newaxis, np.newaxis, :],
                size=(self.batch_size, self.candidate_goals,
                      self.action_dim)).clip(-action_scale.cpu().numpy(),
                                             action_scale.cpu().numpy())).type(
                                                 torch.float32).to(DEVICE)

        candidates = torch.cat([original_goal, diff_goal, random_goals], dim=1)

        # seq_len = states.shape[1]
        seq_len = states.shape[1]

        new_batch_size = seq_len * self.batch_size

        action_dim = actions.shape[2]

        obs_dim = self.state_dim

        ncands = candidates.shape[1]  # 10

        true_actions = actions.reshape((new_batch_size, action_dim))

        observations = states.reshape((new_batch_size, obs_dim))
        goal_shape = (new_batch_size, self.action_dim)

        policy_actions = torch.zeros(
            (ncands, new_batch_size, action_dim)).to(DEVICE)

        for c in range(ncands):
            subgoal = candidates[:, c]
            # candidate = (subgoal + states[:, 1, :self.action_dim]
            #              ).unsqueeze(1) - states[:, :, :self.action_dim]
            candidate = subgoal

            candidate = candidate.repeat_interleave(seq_len, dim=0)
            policy_actions[c] = low_network.policy(observations,
                                                   candidate).detach()

        difference = (policy_actions - true_actions).cpu().numpy()
        # difference = np.where(difference != -np.inf, difference, 0)
        # difference = difference.reshape((ncands, self.batch_size, seq_len,
        #  action_dim)).transpose(1, 0, 2, 3)

        logprob = -0.5 * np.sum(np.linalg.norm(difference, axis=-1)**2,
                                axis=-1)
        assert logprob.shape == (ncands, )
        max_indices = np.argmax(logprob, axis=-1)

        return candidates[:, max_indices]

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

    def pertub(self, act: torch.Tensor):
        mean = torch.zeros_like(act)
        var = torch.ones_like(act)
        return self.action_scale * torch.normal(
            mean, self.expl_noise * var).to(DEVICE)

    def train(self, buffer: ReplayBuffer[Transition], low_con: LowNetwork):
        (states, actions, rewards, n_states, done,
         infos) = resolve_transitions(buffer.sample(self.batch_size),
                                      (self.state_dim, ), (self.action_dim, ))

        goals = torch.stack([i['goal'] for i in infos]).detach()
        states_arr = torch.stack([i['state_arr'] for i in infos]).detach()
        actions_arr = torch.stack([i['action_arr'] for i in infos]).detach()

        actions = self.off_policy_correct(low_con, actions, states_arr,
                                          actions_arr).detach()

        not_done = 1 - done
        n_goals = goals

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
