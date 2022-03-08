# %%
import setup
from utils.common import Step, Episode, TransitionGeneric
from torch import nn
import math
from collections import deque
from gym.wrappers import FrameStack
import torch
from utils.preprocess import PreprocessInterface
from utils.algorithm import AlgorithmInterface
from tqdm.autonotebook import tqdm
from torchvision import transforms as T
from utils.agent import Agent
from gym.spaces import Box
from typing import List, Tuple, Literal, Any, Optional, cast, Callable, Union, Iterable
import gym
import numpy.typing as npt
import numpy as np


# %%
RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# %%
env = gym.make("PongDeterministic-v4")
env.seed(RANDOM_SEED)
env.reset()
TOTAL_ACTIONS = env.action_space.n


# %%
TOTAL_ACTIONS

# %%


class SkipFrame(gym.Wrapper):
    def __init__(self, env: gym.Env, skip: int):
        assert skip >= 0

        """Return only every `skip`-th frame"""
        super().__init__(env)

        self.env = env
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        obs = None
        info = None

        for _ in range(self._skip + 1):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class PreprocessObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.observation_space = Box(
            low=0,
            high=1,
            shape=(1, 84, 84),
            dtype=np.float32,
        )

        self.transform = T.Compose(
            [T.ToPILImage(), T.Resize((84, 84)), T.Grayscale(), T.ToTensor()]
        )

    def observation(self, observation):
        observation = self.transform(observation)
        assert observation.shape == self.observation_space.shape
        return observation


class ToTensorObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        # self.obs_shape = env.observation_space.shape[:2]
        (h, w, c) = env.observation_space.shape
        self.observation_space = Box(
            low=0,
            high=1,
            shape=(c, h, w),
            dtype=np.float32,
        )

        self.transform = T.Compose([T.ToTensor()])

    def observation(self, observation):
        observation = self.transform(observation)
        assert observation.shape == self.observation_space.shape
        return observation


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        (_, h, w) = env.observation_space.shape
        self.observation_space = Box(
            low=0,
            high=1,
            shape=(1,) + (h, w),
            dtype=np.float32,
        )

        self.transform = T.Compose([T.Grayscale()])

    def observation(self, observation):

        observation = self.transform(observation)
        assert observation.shape == self.observation_space.shape
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, _shape: Union[int, Tuple[int, int]]):
        super().__init__(env)

        self.env = env

        if isinstance(_shape, int):
            shape = (_shape, _shape)
        else:
            shape = _shape

        # self.obs_shape = self.observation_space.shape[0:1] + shape
        (c, _, _) = env.observation_space.shape

        self.observation_space = Box(
            low=0,
            high=1,
            shape=(c,) + shape,
            dtype=np.float32,
        )

        self.transforms = T.Compose([T.Resize(shape)])

    def observation(self, observation):
        observation = self.transforms(observation)
        assert observation.shape == self.observation_space.shape
        return observation


# %%
env = PreprocessObservation(env)
# env = GrayScaleObservation(env)
# env = ResizeObservation(env, 84)
env = FrameStack(env, num_stack=4)
env


# %%
# shape is (210, 160, 3)
Observation = torch.Tensor
Action = int

# shape is (4, 210, 160, 3)
State = torch.Tensor
Reward = int

Transition = TransitionGeneric[State, Action]


# %%


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, (8, 8), 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, TOTAL_ACTIONS),
        )

    def forward(self, x: State) -> torch.Tensor:
        rlt = cast(torch.Tensor, self.net(x.to(device)))
        assert rlt.shape == (x.shape[0], TOTAL_ACTIONS)
        return rlt.cpu()


# %%
class RandomAlgorithm(AlgorithmInterface[State, Action]):
    def __init__(self):
        # self.after_step_freq = 1
        # self.need_on_termination = True
        self.frame_skip = 0
        self.reset()

    def reset(self):
        self.times = 1
        self.last_action = None

    def allowed_actions(self, state: State) -> List[Action]:
        return list(range(TOTAL_ACTIONS))

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
        sar: Tuple[State, Action, Reward],
        sa: Tuple[State, Optional[Action]],
    ):
        pass


# %%
DEFAULT_TRAINING_TIMES = 50_00_0000


class NNAlgorithm(AlgorithmInterface[State, Action]):
    def __init__(self, training_times: int = 50_00_0000, gamma: float = 0.99):
        self.frame_skip = 0

        self.times = 0

        self.policy_network = DQN().to(device)
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=1e-4
        )

        self.target_network = DQN().to(device)
        self.target_network.load_state_dict(self.policy_network.state_dict())

        for p in self.target_network.parameters():
            p.requires_grad = False

        self.target_network.eval()

        self.batch_size = 32

        self.update_target = 10000

        self.replay_memory: deque[Transition] = deque(
            maxlen=math.ceil(25_0000))

        self.gamma = gamma
        self.loss_func = torch.nn.MSELoss()

        self.update_times = 4

        self.loss: float = -1.0

    def reset(self):
        pass

    def allowed_actions(self, _: State) -> List[Action]:
        return list(range(TOTAL_ACTIONS))

    def take_action(self, state: State) -> Action:
        rand = np.random.random()
        max_decry_times = 100_0000
        sigma = 1 - 0.95 / max_decry_times * \
            np.min([self.times, max_decry_times])
        if rand < sigma:
            return np.random.choice(self.allowed_actions(state))

        else:
            act_vals: torch.Tensor = self.policy_network(
                self.resolve_lazy_frames(state)
            )
            maxi = torch.argmax(act_vals)
            return cast(int, maxi.item())

    def after_step(
        self,
        sar: Tuple[State, Action, Reward],
        sa: Tuple[State, Optional[Action]],
    ):
        (s, a, r) = sar
        (sn, an) = sa
        self.replay_memory.append((s, a, r, sn, an))

        if self.times != 0 and self.times % (self.update_times) == 0:

            if len(self.replay_memory) >= 5 * self.batch_size:

                batch: List[Transition] = []
                for i in np.random.choice(len(self.replay_memory), self.batch_size):
                    batch.append(self.replay_memory[i])

                self.train(batch)

        if self.times != 0 and self.times % (self.update_target * self.update_times) == 0:
            self.update_target_network()

        self.times += 1

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def resolve_lazy_frames(self, s: State) -> torch.Tensor:
        rlt = torch.cat([s[0], s[1], s[2], s[3]]).unsqueeze(0)
        return rlt

    def train(self, batch: List[Transition]):

        masks = torch.tensor(
            [0 if an is None else 1 for (_, _, _, _, an) in batch],
            dtype=torch.float,
        )

        target = torch.tensor(
            [r for (_, _, r, _, _) in batch], dtype=torch.float
        ) + masks * self.gamma * (torch.max(
            self.target_network(
                torch.cat(
                    [self.resolve_lazy_frames(sn)
                     for (_, _, _, sn, _) in batch]
                )
            ).detach(),
            dim=1,
        )[0])

        # s_next = torch.cat([self.resolve_lazy_frames(sn)
        #                     for (_, _, _, sn, _) in batch])
        # assert s_next.shape == (32, 4, 84, 84)
        # q_next = self.target_network(s_next).detach()

        # assert q_next.shape == (32, TOTAL_ACTIONS)

        # target = torch.tensor(
        #     [self.clip_reward(r) for (_, _, r, _, _) in batch], dtype=torch.float
        # ) + torch.inner(
        #     masks,
        #     self.gamma
        #     * q_next.gather(1, torch.argmax(self.policy_network(s_next), dim=1, keepdim=True)).squeeze(1)
        # )

        assert target.shape == (32,)
        s_curr = torch.cat([self.resolve_lazy_frames(s)
                           for (s, _, _, _, _) in batch])
        assert s_curr.shape == (32, 4, 84, 84)

        x_vals = self.policy_network(s_curr)

        x = x_vals.gather(
            1, torch.tensor([a for (_, a, _, _, _) in batch]).unsqueeze(1)
        ).squeeze(1)

        assert x.shape == (32,)

        loss = self.loss_func(x, target)
        self.loss = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_network.parameters():  # gradient clipping
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def on_termination(self, sar: Tuple[List[State], List[Action], List[Reward]]):
        (s, a, r) = sar
        assert len(s) == len(a) + 1
        assert len(s) == len(r) + 1
        pass


class DDQNAlgorithm(NNAlgorithm, AlgorithmInterface[State, Action]):
    def __init__(self, training_times: int = 50_00_0000, gamma: float = 0.99):
        super().__init__(training_times, gamma)

    def train(self, batch: List[Transition]):

        s_next = torch.cat([self.resolve_lazy_frames(sn)
                            for (_, _, _, sn, _) in batch])
        assert s_next.shape == (32, 4, 84, 84)

        q_next = self.target_network(s_next).detach()

        assert q_next.shape == (32, TOTAL_ACTIONS)

        masks = torch.tensor(
            [0 if an is None else 1 for (_, _, _, _, an) in batch],
            dtype=torch.float,
        )

        target = torch.tensor(
            [r for (_, _, r, _, _) in batch], dtype=torch.float
        ) + masks * self.gamma * q_next.gather(1, torch.argmax(self.policy_network(s_next), dim=1, keepdim=True)).squeeze(1)

        assert target.shape == (32,)
        s_curr = torch.cat([self.resolve_lazy_frames(s)
                           for (s, _, _, _, _) in batch])
        assert s_curr.shape == (32, 4, 84, 84)

        x_vals = self.policy_network(s_curr)

        x = x_vals.gather(
            1, torch.tensor([a for (_, a, _, _, _) in batch]).unsqueeze(1)
        ).squeeze(1)

        assert x.shape == (32,)

        loss = self.loss_func(x, target)
        self.loss = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_network.parameters():  # gradient clipping
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


class Preprocess(PreprocessInterface[Observation, Action, State]):
    def __init__(self):
        self.reset()

    def reset(self):
        pass

    def get_current_state(self, h: List[Observation]) -> State:
        assert len(h) > 0

        assert h[-1].shape == (4, 1, 84, 84)
        return h[-1]


# %%
TRAINING_TIMES = DEFAULT_TRAINING_TIMES
# TRAINING_TIMES = 2_0000
# env._max_episode_steps = 1_000

agent = Agent(env, DDQNAlgorithm(TRAINING_TIMES), Preprocess())
training_rwds: List[int] = []

max_decry_times = 100_0000
with tqdm(total=DEFAULT_TRAINING_TIMES) as pbar:
    # for _ in pbar:
    frames = 0
    # pbar.update(1)
    # pbar.update(1)
    # frames = 1
    while frames < TRAINING_TIMES:
        agent.reset(["preprocess"])
        # frames += 1
        i = 0
        end = False
        # while frames < TRAINING_TIMES:
        while not end and frames < TRAINING_TIMES:

            # end = False

            # while not end and frames < TRAINING_TIMES:
            (_, end) = agent.step()
            # pbar.update(1)
            i += 1

            # frames += 1
            # pbar.update(1)
        frames += i
        pbar.update(i)

        sigma = 1 - 0.95 / max_decry_times * \
            np.min([agent.algm.times, max_decry_times])

        training_rwds.append(np.sum([r for r in agent.episode_reward]))
        pbar.set_postfix(
            rwd=training_rwds[-1],
            sigma=sigma,
            memory_ratio=len(agent.algm.replay_memory) / 25_0000,
            loss=agent.algm.loss,
        )

        if frames >= 3_00_0000:
            print("reached 3_00_0000 frames, end!")
            break


# %%
np.save("./training_rwds.arr", np.asarray(training_rwds))
torch.save(agent.algm.policy_network.state_dict(), "./policy_network.params")
torch.save(agent.algm.target_network.state_dict(), "./target_network.params")
# np.save("./training_loss.arr", np.asarray(agent.algm.loss))


# %%
EVALUATION_TIMES = 30
MAX_EPISODE_LENGTH = 18_000
rwds: List[int] = []
agent.toggleEval(True)

for _ in tqdm(range(EVALUATION_TIMES)):
    agent.reset(['preprocess'])

    end = False
    i = 1

    while not end and i < MAX_EPISODE_LENGTH:
        (o, end) = agent.step()
        i += 1
        env.render()
        # if end:
        #     rwds.append(np.sum([r if r is not None else 0 for (_,
        #                                                        _, r) in cast(Episode, episode)]))
    rwds.append(
        np.sum([r for r in agent.episode_reward])
    )

np.save("./eval_rwds.arr", np.asarray(rwds))
