# %%
import gym
import numpy as np
import numpy.typing as npt

from typing import List, Tuple, Literal, Any, Optional, cast, Callable
from utils.agent import Agent
from tqdm.autonotebook import tqdm
from utils.algorithm import AlgorithmInterface
from utils.preprocess import PreprocessInterface
import torch
from torchvision import transforms
from torch import nn
from utils.common import Step, Episode, TransitionGeneric


# %%
RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# %%
env = gym.make("StarGunner-v0")
env.seed(RANDOM_SEED)
env.reset()
print(env.action_space)
env._max_episode_steps = 1_8000

# %%
# shape is (210, 160, 3)
Observation = npt.NDArray[np.uint8]
Action = int

# shape is (4, 210, 160, 3)
State = torch.Tensor
Reward = int

Transition = TransitionGeneric[State, Action]


# %%
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # self.first = nn.Sequential(nn.Conv2d(4, 32, (8, 8), 4), nn.ReLU()) self.second = nn.Sequential()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, (8, 8), 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.Linear(512, 18),
        ).to(device)

    def forward(self, x: State) -> torch.Tensor:
        rlt = cast(torch.Tensor, self.net(x.to(device)))
        return rlt


# %%
class RandomAlgorithm(AlgorithmInterface[State, Action]):
    def __init__(self):

        self.times = 0
        self.last_action = None

    def allowed_actions(self, state: State) -> List[Action]:
        return list(range(18))

    def take_action(self, state: State) -> Action:
        self.times += 1

        if self.times % 10 == 0:
            act = np.random.choice(self.allowed_actions(state))
            self.last_action = act
            return act

        if self.last_action is not None:
            return self.last_action

        act = np.random.choice(self.allowed_actions(state))
        self.last_action = act
        return act

    def after_step(
        self,
        sa: Tuple[State, Action],
        episode: Episode[State, Action],
    ):
        pass

    def on_termination(self, episode: Episode[State, Action]):
        pass


class NNAlgorithm(AlgorithmInterface[State, Action]):
    def __init__(self, nn: DQN, sigma: float, gamma: float = 0.95):
        self.network = nn
        self.sigma = sigma

        self.times: int = 0
        self.batch_size = 32

        self.memory_replay: List[Transition] = []
        self.gamma = gamma
        self.loss_func = torch.nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(
            self.network.parameters(), 1e-3, 0.95, 0.95, 1e-2
        )

    def allowed_actions(self, state: State) -> List[Action]:
        return list(range(18))

    def take_action(self, state: State) -> Action:
        self.times += 1
        rand = np.random.random()
        sigma = self.sigma * (-0.9 / 100_0000 * self.times + 1)
        if rand < sigma:
            return np.random.choice(self.allowed_actions(state))
        else:
            act_vals: torch.Tensor = self.network(state)
            maxi = torch.argmax(act_vals)
            return cast(Action, maxi)

    def after_step(
        self,
        sa: Tuple[State, Action],
        episode: Episode[State, Action],
    ):
        pass

    def train(self, batch: List[Transition]):
        # batch = torch.tensor(_b)
        target = torch.tensor(
            [
                r if an is None else r + self.gamma * torch.max(self.network(sn))
                for (s, a, r, sn, an, _) in batch
            ]
        )
        assert target.shape == (32,)
        x = torch.cat([self.network(s)[:, a] for (s, a, r, _, _, _) in batch])
        # x = self.network(torch.cat([s for (s, _, _, _, _, _) in batch]))[:, 0]
        loss = self.loss_func(x, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        pass

    def extract_transitions(self, episode: Episode[State, Action]) -> List[Transition]:
        trs: List[Transition] = []
        for (idx, (s, a, r)) in enumerate(episode[:-1]):
            (sn, an, rn) = episode[idx + 1]
            trs.append((s, cast(Action, a), cast(float, r), sn, an, rn))
        return trs

    def on_termination(self, episode: Episode[State, Action]):
        trs = self.extract_transitions(episode)
        for tr in trs:
            if len(self.memory_replay) < 100_0000:
                self.memory_replay.append(tr)
            else:
                self.memory_replay.pop()
                self.memory_replay.append(tr)

        if len(self.memory_replay) <= 48:
            pass

        # self.train(np.random.choice(
        #     np.asarray(self.memory_replay), 32).tolist())
        # self.train((np.asarray(self.memory_replay)[ np.random.randint(0, len(self.memory_replay), 32)]).tolist())
        batch: List[Transition] = []
        # batch = (np.asarray(self.memory_replay))[
        #     np.random.randint(0, len(self.memory_replay), 32)
        # ]
        for i in np.random.randint(0, len(self.memory_replay), 32):
            batch.append(self.memory_replay[i])

        self.train(batch)


class Preprocess(PreprocessInterface[Observation, Action, State]):
    def __init__(self):
        self.trfm: Callable[[Observation], State] = transforms.Compose(
            [transforms.ToTensor(), transforms.Grayscale(), transforms.Resize((84, 84))]
        )
        self.history: Episode[State, Action] = []

    def reset(self):
        self.history = []

    def get_current_state(self, h: Episode[Observation, Action]) -> State:
        assert len(h) > 0

        last_4_arr = self.stack_4(h, -1)

        rlt = torch.stack([self.trfm(i) for i in last_4_arr]).squeeze(1).unsqueeze(0)
        assert rlt.shape == (1, 4, 84, 84)
        return rlt

    def stack_4(self, h: Episode[Observation, Action], idx: int) -> npt.NDArray:

        assert idx < 0
        last_4_index = [-12 + idx, -8 + idx, -4 + idx, idx]

        last_4: List[Observation] = []
        for idx in last_4_index:
            if -idx <= len(h):
                last_4.append(np.asarray((h[idx][0])))

        last_4_arr = np.asarray(last_4)
        while last_4_arr.shape[0] < 4:
            last_4_arr = np.insert(last_4_arr, 0, last_4[0], axis=0)

        assert last_4_arr.shape == (4, 210, 160, 3)

        return last_4_arr

    def transform_history(
        self, h: Episode[Observation, Action]
    ) -> Episode[State, Action]:
        assert len(h) == len(self.history) + 1

        (o, a, r) = h[-1]
        last_4_arr = self.stack_4(h, -1)
        # s = torch.stack
        s = torch.stack([self.trfm(i) for i in last_4_arr]).squeeze(1).unsqueeze(0)
        assert s.shape == (1, 4, 84, 84)
        self.history.append((s, a, r))

        # if len(h) > l:
        #     self.history.extend([(self.trfm(o), a, r) for (o, a, r) in h[l:]])
        #     return self.history
        # else:
        return self.history


# %%
agent = Agent(env, NNAlgorithm(DQN(), 1e-3, 0.95), Preprocess())


# %%
# TRAINING_TIMES = 50_00_0000


# frames = 1
# while frames < TRAINING_TIMES:
#     agent.reset()

#     end = False

#     while not end:
#         (o, end, episode) = agent.step()
#         frames += 1


# %%
EVALUATION_TIMES = 30
MAX_EPISODE_LENGTH = 18_000

rwds: List[int] = []

for _ in tqdm(range(EVALUATION_TIMES)):
    agent.reset()

    end = False
    i = 1

    while not end and i < MAX_EPISODE_LENGTH:
        (o, end, episode) = agent.step()
        i += 1
        # env.render()
        # if end:
        #     rwds.append(np.sum([r if r is not None else 0 for (_,
        #                                                        _, r) in cast(Episode, episode)]))
    rwds.append(np.sum([r if r is not None else 0 for (_, _, r) in agent.episode]))


# %%
np.mean(rwds)
