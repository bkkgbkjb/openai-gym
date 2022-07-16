from inspect import GEN_CLOSED
import setup
from utils.algorithm import (ActionInfo, Mode, ReportInfo)
from utils.common import Info
from utils.episode import Episode
from utils.step import NotNoneStep, Step
from utils.transition import (Transition, resolve_transitions, TransitionTuple)
from torch import Tensor, nn
from collections import deque
import torch
from utils.preprocess import PreprocessI
from utils.algorithm import Algorithm
from torch.distributions import Categorical, Normal
from typing import Union
from utils.nets import NeuralNetworks, layer_init
from torch.utils.data import DataLoader
from args import args

from typing import List, Tuple, Any, Optional, Callable, Dict, cast
import numpy as np

from utils.replay_buffer import ReplayBuffer

State = torch.Tensor
O = torch.Tensor
Action = torch.Tensor

S = O
Reward = int

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Goal = torch.Tensor


class Preprocess(PreprocessI[O, S]):

    def __init__(self):
        pass

    def on_agent_reset(self):
        pass

    def get_current_state(self, h: List[O]) -> S:
        assert len(h) > 0

        # assert h[-1].shape == (4, 1, 84, 84)
        return torch.from_numpy(h[-1]).type(torch.float32).to(DEVICE)


class SAEncoder(NeuralNetworks):

    def __init__(self, n_state: int, n_action: int, repr_dim: int = 16):
        super().__init__()
        self.n_state = n_state
        self.n_action = n_action
        self.n_repr = repr_dim

        self.net = nn.Sequential(
            layer_init(nn.Linear(self.n_state + self.n_action, 1024)),
            nn.ReLU(), layer_init(nn.Linear(1024, 1024)), nn.ReLU(),
            layer_init(nn.Linear(1024, self.n_repr)))

    def forward(self, state: State, action: Action):
        return self.net(torch.cat([state, action], dim=1))


class GEncoder(NeuralNetworks):

    def __init__(self, n_goal: int, repr_dim: int = 16):
        super().__init__()
        self.n_goal = n_goal
        self.n_repr = repr_dim

        self.net = nn.Sequential(layer_init(nn.Linear(self.n_goal, 1024)),
                                 nn.ReLU(), layer_init(nn.Linear(1024, 1024)),
                                 nn.ReLU(),
                                 layer_init(nn.Linear(1024, self.n_repr)))

    def forward(self, state: State):
        return self.net(state)


class QFunction(NeuralNetworks):

    def __init__(self, n_state: int, n_action: int, n_goal: int):
        super().__init__()
        self.sa_encoder = SAEncoder(n_state, n_action)
        self.g_encoder = GEncoder(n_goal)

        self.n_state = n_state
        self.n_action = n_action
        self.n_goal = n_goal

    def forward(
            self, s: State, a: Action, goal: Goal
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        sa_value = self.sa_encoder(s, a)
        g_value = self.g_encoder(goal)
        return torch.einsum('bi,bi->b', sa_value, g_value), (sa_value, g_value)


class PaiFunction(NeuralNetworks):

    def __init__(self, n_state: int, n_action: int, n_goal: int,
                 action_scale: float):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(n_state + n_goal, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.ReLU(),
        ).to(DEVICE)

        self.mean = layer_init(nn.Linear(1024, n_action)).to(DEVICE)
        self.std = layer_init(nn.Linear(1024, n_action)).to(DEVICE)

        self.n_state = n_state
        self.n_action = n_action
        self.action_scale = action_scale
        self.n_goal = n_goal

    def forward(self, s: State, g: Goal) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(torch.cat([s.to(DEVICE), g.to(DEVICE)], dim=1))
        mean = self.mean(x)
        log_std = self.std(x)
        log_std = torch.clamp(log_std, min=-12, max=6)

        assert mean.shape == (s.size(0), self.n_action)
        assert log_std.shape == (s.size(0), self.n_action)

        return mean, log_std

    def sample(self, s: State, g: Goal):
        mean, log_std = self.forward(s, g)
        std = log_std.exp()
        normal = Normal(mean, std)
        raw_act = normal.rsample()
        assert raw_act.shape == (s.size(0), self.n_action)

        y_t = torch.tanh(raw_act)
        act = y_t * self.action_scale

        raw_log_prob = normal.log_prob(raw_act)
        assert raw_log_prob.shape == (s.size(0), self.n_action)

        mod_log_prob = (self.action_scale * (1 - y_t.pow(2)) + 1e-6).log()
        assert mod_log_prob.shape == (s.size(0), self.n_action)

        log_prob = (raw_log_prob - mod_log_prob).sum(1, keepdim=True)

        mean = self.action_scale * torch.tanh(mean)
        return act, log_prob, mean, normal


class CRL(Algorithm):

    def __init__(
        self,
        n_state: int,
        n_goals: int,
        n_actions: int,
        action_scale: float,
    ):
        self.set_name("CRL")
        self.n_actions = n_actions
        self.n_state = n_state
        self.n_goals = n_goals

        self.gamma = 0.99
        self.action_scale = action_scale

        self.tau = 5e-3

        self.start_traininig_size = int(1e4)
        self.mini_batch_size = 1024

        self.policy = PaiFunction(self.n_state, self.n_actions, self.n_goals,
                                  self.action_scale).to(DEVICE)

        self.q1 = QFunction(self.n_state, self.n_actions,
                            self.n_goals).to(DEVICE)
        self.q2 = QFunction(self.n_state, self.n_actions,
                            self.n_goals).to(DEVICE)

        self.q1_target = self.q1.clone().no_grad()
        self.q2_target = self.q2.clone().no_grad()

        self.q1_loss = nn.BCEWithLogitsLoss()
        self.q2_loss = nn.BCEWithLogitsLoss()

        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), 3e-4)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), 3e-4)

        self.p_optimizer = torch.optim.Adam(self.policy.parameters(), 3e-4)
        self.c = 15
        self.future_step_idx_dist = torch.distributions.geometric.Geometric(
            1 - self.gamma)

        self.reset()

    def reset(self):
        self.episodes = None
        self.replay_memory = ReplayBuffer[Episode[State]](None)
        self.reset_episode_info()

    def reset_episode_info(self):
        self.inner_steps = 0
        self.fg = None

    def on_env_reset(self, mode: Mode, info: Dict[str, Any]):
        assert mode == 'eval'

        assert self.fg is None
        env = info['env']

        self.fg = torch.tensor(env.target_goal,
                               dtype=torch.float32,
                               device=DEVICE)

    @torch.no_grad()
    def take_action(self, mode: Mode, state: State) -> Action:
        assert self.fg is not None
        action, _, max_actions, _ = self.policy.sample(state.unsqueeze(0),
                                                       self.fg.unsqueeze(0))

        return (max_actions if mode == 'eval' else action).squeeze()

    def on_episode_termination(
        self, mode: Mode, sari: Tuple[List[S], List[Action], List[Reward],
                                      List[Info]]
    ) -> Optional[ReportInfo]:
        (_, _, r, _) = sari
        report = {'success_rate': 1 if any([_r >= 0.7 for _r in r]) else 0}

        self.reset_episode_info()
        return report

    def train(self):
        # (states, actions, _, _, _, infos) = resolve_transitions(
        #     self.replay_memory.sample(self.mini_batch_size), (self.n_state, ),
        #     (self.n_actions, ))
        episode_sampled = self.replay_memory.sample(self.mini_batch_size)
        L = episode_sampled[0].len
        step_idx = np.random.choice(L, self.mini_batch_size)

        steps = [e.steps[step_idx[i]] for i, e in enumerate(episode_sampled)]

        states = torch.stack([s.state for s in steps]).to(DEVICE)

        assert states.shape == (self.mini_batch_size, self.n_state)
        # actions = torch.stack(
        #     [NotNoneStep.from_step(s).action for (s, _) in steps]).to(DEVICE)

        actions = torch.stack([NotNoneStep.from_step(s).action
                               for s in steps]).to(DEVICE)

        assert actions.shape == (self.mini_batch_size, self.n_actions)

        goals = torch.stack([s.info['future_state'] for s in steps]).to(DEVICE)
        # goals = torch.stack([i['future_state'] for i in infos]).to(DEVICE)
        assert goals.shape == (self.mini_batch_size, self.n_goals)

        # final_goals = torch.stack([
        #     torch.from_numpy(s.info['goal']).type(torch.float32)
        #     for (s, _) in steps
        # ]).to(DEVICE)

        # future_goals = torch.stack([s.state[:2]
        #                             for (_, s) in steps]).to(DEVICE)

        # assert final_goals.shape == future_goals.shape == (
        #     self.mini_batch_size, self.n_goals)

        (_, (sa_1, g_1)) = self.q1(states, actions, goals)
        assert sa_1.shape == g_1.shape == (self.mini_batch_size, 16)

        predicted_q1 = torch.inner(sa_1, g_1)
        assert predicted_q1.shape == (self.mini_batch_size,
                                      self.mini_batch_size)

        (_, (sa_2, g_2)) = self.q2(states, actions, goals)
        predicted_q2 = torch.inner(sa_2, g_2)

        q_val_loss1 = self.q1_loss(predicted_q1,
                                   torch.eye(self.mini_batch_size).to(DEVICE))
        q_val_loss2 = self.q2_loss(predicted_q2,
                                   torch.eye(self.mini_batch_size).to(DEVICE))

        self.q1_optimizer.zero_grad()
        q_val_loss1.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q_val_loss2.backward()
        self.q2_optimizer.step()

        # Training P Function
        new_actions, new_log_probs, _, normal = self.policy.sample(
            states, goals)
        (critic_1, _) = self.q1(states, new_actions, goals)
        assert critic_1.shape == (self.mini_batch_size, )
        (critic_2, _) = self.q2(states, new_actions, goals)

        policy_loss = -torch.min(critic_1, critic_2).mean()
        bc_loss = -(normal.log_prob(actions)).mean()

        actor_loss = (0.95 * policy_loss + 0.05 * bc_loss)

        self.p_optimizer.zero_grad()
        actor_loss.backward()
        # bc_loss.backward()
        self.p_optimizer.step()

        self.report(
            dict(
                policy_loss=actor_loss,
                q_loss1=q_val_loss1,
                q_loss2=q_val_loss2,
            ))

    def get_episodes(self, episodes: List[Episode]):
        for e in episodes:
            l = e.len
            arange = torch.arange(l + 1)

            is_future_mask = arange[:, None] < arange[None]
            discount = self.gamma**(arange[None] - arange[:, None])
            probs = is_future_mask * discount
            goal_index = torch.distributions.Categorical(
                logits=torch.log(probs + 1e-7)).sample()
            assert goal_index.shape == (l + 1, )

            # state = torch.stack([s.state for s in e.steps])
            # next_state = torch.stack([s.info['next'].state for s in e.steps])

            goal = torch.stack([s.state[:2] for s in e._steps])

            goal = goal[goal_index[:-1]]

            assert goal.shape == (l, 2)

            for j, s in enumerate(e.steps):
                # s = NotNoneStep.from_step(s)
                s.add_info('future_state', goal[j])

                # self.replay_memory.append(
                #     Transition((NotNoneStep(s.state, s.action, s.reward,
                #                             s.info),
                #                 Step(s.info['next'].state, None, None,
                #                      s.info['next'].info))))
                if j == l - 1:
                    assert s.info['next'].is_end()
            self.replay_memory.append(e)
            # transition = Transition(())

    def manual_train(self, info: Dict[str, Any]):
        assert 'episodes' in info
        if self.episodes != info['episodes']:
            assert self.episodes is None
            self.episodes = info['episodes']
            self.get_episodes(self.episodes)

        self.train()
        return self.mini_batch_size