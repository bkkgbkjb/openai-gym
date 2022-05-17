import gym
from typing import (
    List,
    Tuple,
    Dict,
    Literal,
    Any,
    Optional,
    cast,
    Callable,
    Union,
    Iterable,
    TypeVar,
)
from tqdm import tqdm
from gym.wrappers import FrameStack, LazyFrames
from os import times
from gym.spaces import Box
import numpy as np
from torchvision import transforms as T
import torch

from utils.agent import Agent


def resolve_lazy_frames(lazy_frames: Any) -> torch.Tensor:
    assert len(lazy_frames) == 4
    rlt = torch.cat(
        cast(
            List[torch.Tensor],
            [lazy_frames[0], lazy_frames[1], lazy_frames[2], lazy_frames[3]],
        )
    ).unsqueeze(0)
    assert rlt.shape == (1, 4, 84, 84)
    return rlt


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


class ToTensorEnv(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        (h, w, c) = env.observation_space.shape

        self.observation_space = Box(low=0, high=1, dtype=np.float32, shape=(c, h, w))

        self.transform = T.ToTensor()

    def observation(self, observation):
        observation = self.transform(observation)
        assert observation.shape == self.observation_space.shape
        return observation


def glance(env: gym.Env, random_seed=0, repeats=3):
    env.seed(random_seed)
    env.action_space.seed(random_seed)
    env.observation_space.seed(random_seed)
    env.reset()

    print(env.action_space, env.observation_space)
    for _ in range(repeats):
        r = 0.0
        env.reset()
        s = False
        t = 0

        while not s:
            env.render(mode="human")

            times.sleep(100)
            (_, rwd, stop, _) = env.step(env.action_space.sample())
            t += 1

            r += rwd

            if stop:
                print(f"rwd is: {r}, total steps: {t}")
                break


A = TypeVar("A")
S = TypeVar("S")
O = TypeVar("O")


def train(
    env: gym.Env, agent: Agent[O, S, A], random_seed=0, training_frames: int = int(1e6)
) -> Agent[O, S, A]:
    env.seed(random_seed)
    env.action_space.seed(random_seed)
    env.observation_space.seed(random_seed)
    env.reset()

    TRAINING_TIMES = training_frames

    print(env.action_space, env.observation_space)
    agent.reset()
    print(f"train agent: {agent.name}")

    agent.toggleEval(False)

    with tqdm(total=TRAINING_TIMES) as pbar:
        frames = 0
        epi = 0
        while frames < TRAINING_TIMES:
            agent.reset()
            i = 0
            end = False
            while not end and frames < TRAINING_TIMES:

                (_, end) = agent.step()
                i += 1

            epi += 1
            frames += i

            pbar.update(i)

        print(f"training end after {frames} frames")

    return agent


def eval(agent: Agent[O, S, A], repeats=10) -> Agent[O, S, A]:
    agent.toggleEval(True)

    print('eval process start')
    for _ in range(repeats):
        agent.reset()

        while True:
            # agent.render(mode="human")

            (_, s) = agent.step()

            if s:
                break

    print('eval process end')
    return agent
