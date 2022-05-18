import gym
import math
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


def train(agent: Agent[O, S, A], training_frames=int(1e6)) -> Agent[O, S, A]:

    agent.reset()

    agent.toggleEval(False)

    with tqdm(total=training_frames) as pbar:
        frames = 0
        while frames < training_frames:
            agent.reset()
            i = 0
            end = False
            while not end and frames < training_frames:

                (_, end) = agent.step()
                i += 1

            frames += i

            pbar.update(i)


    return agent


def eval(agent: Agent[O, S, A], repeats=10) -> Agent[O, S, A]:
    agent.toggleEval(True)

    for _ in range(repeats):
        agent.reset()

        while True:

            (_, s) = agent.step()

            if s:
                break

    return agent


def train_and_eval(
    agent: Agent[O, S, A],
    single_train_frames=int(1e4),
    eval_repeats=10,
    total_train_frames=int(1e6),
) -> Agent[O, S, A]:

    for _ in tqdm(range(math.ceil(total_train_frames / single_train_frames))):
        train(agent, single_train_frames)
        eval(agent, eval_repeats)

    return agent
