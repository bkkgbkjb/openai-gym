# %%
import numpy as np

# from gym.envs.toy_text import BlackjackEnv
import gym
from typing import Literal, List, Tuple, cast, Dict, Optional, Callable, Protocol, Union
import plotly.graph_objects as go
from copy import deepcopy
from abc import abstractmethod, ABC
import math
import sys
from primefac import primegen
from tqdm.autonotebook import tqdm
import plotly.express as px

import io


# %%
RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)


# %%
env = gym.make('MountainCar-v0')
env.seed(RANDOM_SEED)
env


# %%
env.reset()

# %%
# env.render()

# %%
Position = float
Velocity = float
State = Tuple[Position, Velocity]

Action = Literal[0, 1, 2]

StateAction = Tuple[State, Action]
Observation = State

"""
- 0: Accelerate to the Left
- 1: Don't accelerate
- 2: Accelerate to the Right
"""
Reward = float
Step = Tuple[State, Optional[Action], Optional[Reward]]
Episode = List[Step]

# all_states: List[State] = list(range(1000))
all_actions: List[Action] = [0, 1, 2]
# nums_of_all_state = len(all_states)
# nums_of_all_state_action = len(all_states) * len(all_actions)
# allowed_actions: List[List[Action]] = [
#     all_actions for _ in range(nums_of_all_state)]


Feature = np.ndarray
Weight = np.ndarray


# %%
class FeatureInterface(Protocol):
    len: int

    @abstractmethod
    def to_feature(self, sa: StateAction) -> Feature:
        raise NotImplementedError()

    def one_hot_encode(self, a: Action) -> List[int]:
        assert a in all_actions, f"bad action encountered: {a}"
        if a == 0:
            return [1, 0, 0]
        elif a == 1:
            return [0, 1, 0]
        else:
            return [0, 0, 1]


class Tiling(FeatureInterface):
    def __init__(self, n_tilings: int, input_range: List[Tuple[float, float]]):

        if n_tilings // 2 == 0:
            n_tilings = n_tilings + 1

        assert n_tilings > 1, f"number of tilings cannot be lower than 2: {n_tilings}"

        self.input_range = input_range
        self.dimension = len(input_range)

        assert self.dimension >= 1, f"cannot manupilate on 0-dimension"

        self.n_tilings = n_tilings
        # self.n_tiles_per_tiling = n_tilings
        # self.len = n_tilings * (2 * n_tilings - 1)
        self.len = n_tilings * n_tilings * self.dimension + 3

        self.tilings = self.compute_tilings()

    def get_tiling_point(self, nums: List[float], n_partitions: int) -> List[float]:
        assert len(nums) >= 2, f"unexpected nums encountered: {nums}"
        assert n_partitions >= 2, f"unexpected n_partitions encountered: {n_partitions}"

        (l, r) = (nums[0], nums[-1])

        delta = (r - l) / n_partitions

        return [l, *[l + i * delta for i in range(1, n_partitions)], r]

    def find_lowest_prime(self) -> int:
        return 13
        return list(primegen(100 / self.n_tilings))[-1]

    def compute_tilings(self) -> List[List[List[float]]]:
        points: List[List[List[float]]] = []

        for (l, r) in self.input_range:
            points_one_dimen: List[List[float]] = []

            pivot_points = self.get_tiling_point([l, r], self.n_tilings)
            points_one_dimen.append(pivot_points)

            move_delta = self.find_lowest_prime()

            # print(int(self.n_tilings + 1) / 2)
            for i in range(1, int((self.n_tilings + 1) / 2)):
                points_one_dimen.append(
                    self.move_tiling_points(pivot_points, move_delta * i)
                )
                points_one_dimen.append(
                    self.move_tiling_points(pivot_points, -1 * move_delta * i)
                )

            points.append(points_one_dimen)

        return points

    def point_in_range(self, p: float, rang: List[float]) -> List[Literal[0, 1]]:
        v: List[Literal[0, 1]] = []
        for i in range(1, len(rang)):
            if rang[i - 1] <= p < rang[i]:
                v.append(1)
            else:
                v.append(0)

        assert (
            len(v) == len(rang) - 1
        ), f"bad length of v encountered: {len(v)}, {len(rang)-1}"

        return v

    def to_feature(self, sa: StateAction) -> Feature:
        ((pos, vel), a) = sa
        pos_dim = self.tilings[0]
        vel_dim = self.tilings[1]

        pos_feat = [self.point_in_range(pos, t) for t in pos_dim]
        vel_feat = [self.point_in_range(vel, t) for t in vel_dim]

        pos_feat_one_dim = [inner for outer in pos_feat for inner in outer]
        vel_feat_one_dim = [inner for outer in vel_feat for inner in outer]

        return np.asarray(
            [*pos_feat_one_dim, *vel_feat_one_dim, *self.one_hot_encode(a)]
        )

    def move_tiling_points(
        self,
        points: List[float],
        percentile: int,
    ) -> List[float]:
        # return [(np.round(lp + delta), np.round(rp + delta)) for (lp, rp) in tiling]
        # delta = splitting_points[1] - splitting_points[0]
        assert len(points) >= 2, f"bad points: {points}"

        rang = points[-1] - points[0]

        return [p + percentile / 100 * rang for p in points]


class TestFeature(FeatureInterface):
    def __init__(self):
        self.len = 8

    def to_feature(self, sa: StateAction) -> Feature:
        ((pos, vel), a) = sa
        v = [1, pos, vel, pos + vel, pos * vel, *self.one_hot_encode(a)]
        assert len(v) == self.len, f"unexpected length encountered: {len(v)}"
        return np.asarray(v)


class AppxInterface(Protocol):
    feature_algorithm: FeatureInterface

    @abstractmethod
    def predict(self, sa: StateAction, w: Weight) -> float:
        raise NotImplemented()

    @abstractmethod
    def gradient(self, sa: StateAction, w: Weight) -> np.ndarray:
        raise NotImplemented()


class Linear(AppxInterface):
    def __init__(self, feature_algorithm: FeatureInterface):
        self.feature_algorithm = feature_algorithm

    def predict(self, sa: StateAction, w: Weight) -> float:
        return np.inner(self.feature_algorithm.to_feature(sa), w)

    def gradient(self, sa: StateAction, w: Weight) -> np.ndarray:
        return self.feature_algorithm.to_feature(sa)


class PolicyInterface(Protocol):
    appx_algorithm: AppxInterface

    @abstractmethod
    def allowed_actions(self, s: State) -> List[Action]:
        raise NotImplementedError()

    @abstractmethod
    def take_action(self, s: State, w: Weight) -> Action:
        raise NotImplementedError()


class SigmaGreddy(PolicyInterface):
    def __init__(self, sigma: float, appx_algorithm: AppxInterface):
        assert 0 <= sigma <= 1, "sima out of bound"
        self.sigma = sigma
        self.appx_algorithm = appx_algorithm

    def allowed_actions(self, s: State) -> List[Action]:
        return all_actions

    def take_action(self, s: State, w: Weight) -> Action:
        rand = np.random.random()
        all_actions = self.allowed_actions(s)
        if rand < self.sigma:
            return np.random.choice(all_actions)
        else:
            maxi = np.argmax(
                [self.appx_algorithm.predict((s, a), w) for a in all_actions]
            )
            return all_actions[maxi]


class AlwaysRight(PolicyInterface):
    def __init__(self, appx_algorithm: AppxInterface):
        self.appx_algorithm = appx_algorithm

    def allowed_actions(self, s: State) -> List[Action]:
        return all_actions

    def take_action(self, s: State, w: Weight) -> Action:
        return 2


class AlgorithmInterface(Protocol):
    n_of_omega: int
    policy_algorithm: PolicyInterface

    @abstractmethod
    def after_step(
        self, cur_state_action: StateAction, episode: Episode, omega: np.ndarray
    ):
        raise NotImplementedError()

    @abstractmethod
    def on_termination(self, episode: Episode, omega: np.ndarray):
        raise NotImplementedError()

    def allowed_actions(self, s: State) -> List[Action]:
        return self.policy_algorithm.allowed_actions(s)

    def is_terminal_state(self, s: State) -> bool:
        (pos, _) = s
        if pos >= 0.5:
            return True
        return False

    def take_action(self, s: State, omega: Weight) -> Action:
        if self.is_terminal_state(s):
            return np.random.choice(self.allowed_actions(s))

        return self.policy_algorithm.take_action(s, omega)

    def predict(self, sa: StateAction, w: Weight) -> float:
        # assert -1 <= s <= len(all_states), f"unexpected state encounter: {s}"
        (s, _) = sa
        if self.is_terminal_state(s):
            return 0

        # if s == -1 or s == len(all_states):
        #     return 0

        return self.policy_algorithm.appx_algorithm.predict(sa, w)

    def gradient(self, sa: StateAction, w: Weight) -> np.ndarray:
        # assert 0 <= s < len(all_states), f"unexpected state encounter: {s}"

        return self.policy_algorithm.appx_algorithm.gradient(sa, w)


class Sarsa(AlgorithmInterface):
    def __init__(
        self,
        alpha: float,
        policy_algorithm=PolicyInterface,
        gamma: float = 1.0,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.policy_algorithm = policy_algorithm

        self.n_of_omega = self.policy_algorithm.appx_algorithm.feature_algorithm.len

    def after_step(
        self, this_state_action: StateAction, episode: Episode, omega: np.ndarray
    ):
        # (this_s, this_a) = this_state_action
        history = episode[-1:]
        gamma = self.gamma

        # if len(history) != (1 + n):
        #     return
        assert len(history) == (
            1
        ), f"unexpected history length encountered: {len(history)}"

        (old_s, old_a, r) = history[0]

        rwd = cast(Reward, r) + gamma * self.predict(this_state_action, omega)

        omega += (
            self.alpha
            * (rwd - self.predict((old_s, cast(Action, old_a)), omega))
            * self.gradient((old_s, cast(Action, old_a)), omega)
        )

    def on_termination(self, episode: Episode, omega: Weight):
        pass


# %%
class Agent:
    def __init__(
        self,
        env: gym.Env,
        algm: AlgorithmInterface,
    ):
        self.env = env
        self.algm = algm
        self.clear()

    def reset(self):
        self.cur_state: State = self.env.reset()
        self.ready_act: Optional[Action] = None
        self.end = False
        self.episode: Episode = []

    def clear(self):
        self.reset()

        self.omega = np.asarray(
            # [np.random.random() for _ in range(self.algm.n_of_omega)]
            [0.0 for _ in range(self.algm.n_of_omega)]
        )
        # self.episodes: List[Episode] = []

    def step(self) -> Tuple[Observation, bool, Optional[Episode]]:
        assert not self.end, "cannot step on a ended agent"

        act = self.ready_act or self.algm.take_action(self.cur_state, self.omega)
        (obs, rwd, stop, _) = self.env.step(act)
        obs = cast(Observation, obs)

        self.episode.append((self.cur_state, act, rwd))

        self.cur_state = obs

        self.ready_act = self.algm.take_action(self.cur_state, self.omega)

        self.algm.after_step((self.cur_state, self.ready_act), self.episode, self.omega)

        if stop:
            self.episode.append((self.cur_state, None, None))
            # self.episodes.append(self.episode)
            self.end = True
            self.algm.on_termination(self.episode, self.omega)
            # self.episode = []
            return (obs, stop, self.episode)

        return (obs, stop, None)

    def render(self):
        self.env.render()

    def close(self):
        self.clear()
        self.env.close()

    def predict(self, s: State) -> float:
        return np.max(
            [
                self.algm.predict((s, a), self.omega)
                for a in self.algm.allowed_actions(s)
            ]
        )


# %%
TOTAL_TRAINING_EPISODES = 500
MAX_EPISODE_STEPS_IN_TRAINING = 1e3
env._max_episode_steps= MAX_EPISODE_STEPS_IN_TRAINING
agent = Agent(
    cast(gym.Env, env),
    # Sarsa(2e-2, SigmaGreddy(0.1, Linear(TestFeature())))
    # Sarsa(0.3, SigmaGreddy(0.05, Linear(TestFeature())))
    Sarsa(0.1, SigmaGreddy(0, Linear(Tiling(7, [(-1.2, 0.6), (-0.07, 0.07)]))))
    # TDN(9, 2e-4, Linear(), Tiling(5))
)


training = tqdm(range(TOTAL_TRAINING_EPISODES))

# last_omega: Optional[np.ndarray] = None

run_rewards: List[float] = []

for run in training:
    agent.reset()
    end = False
    while not end:
        _, end, episode = agent.step()

        # agent.render()
        

        if end:
            run_rewards.append(
                np.sum(
                    [r if r is not None else 0 for (_, _, r) in cast(Episode, episode)]
                )
            )

    # if run > 1:
    #     progress.set_postfix_str(
    #         str(np.linalg.norm(agent.omega - last_omega)))

    # last_omega = deepcopy(agent.omega)


# %%
agent.omega

# %%
[r for r in run_rewards if r > -1 * MAX_EPISODE_STEPS_IN_TRAINING]


# %%
TOTAL_EVALUATE_TIMES = 100
success_times = 0
MAX_EPISODE_STEPS_IN_EVALUATE = 200
env._max_episode_steps= MAX_EPISODE_STEPS_IN_EVALUATE
evaluate = tqdm(range(TOTAL_EVALUATE_TIMES))
for run in evaluate:
    agent.reset()
    end = False
    i = 0
    while not end:
        _, end, episode = agent.step()

        # agent.render()
        i += 1
        agent.render()

    if i < MAX_EPISODE_STEPS_IN_EVALUATE:
        success_times += 1
        print(f"success: {i}")


# %%
agent.close()

# %%
# fig = go.Figure()
# fig.add_trace(
#     go.Scatter(x=[i + 1 for i in range(len(omegas) - 1)],
#                y=[np.linalg.norm(omegas[i] - omegas[i-1]) for i in range(1, len(omegas))], mode="lines", name="omegas")
# )
# fig.show()


# %%
# agent.episodes[:20]


# %%
# agent.omega

# %%
# true_values = np.load("./true_values_arr.npy", allow_pickle=False)
# true_values


# %%
# fig = go.Figure()
# s = list(range(1000))
# fig.add_trace(
#     go.Scatter(x=[i + 1 for i in s], y=true_values, mode="lines", name="true values")
# )
# fig.add_trace(
#     go.Scatter(
#         x=[i + 1 for i in s],
#         y=[agent.predict(i) for i in s],
#         mode="lines",
#         name="monte-carlo prediction",
#     )
# )
# fig.show()



