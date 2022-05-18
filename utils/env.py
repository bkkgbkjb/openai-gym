import gym
import math
from typing import (
    List,
    Any,
    cast,
    TypeVar,
)
from tqdm import tqdm
from os import times
from gym.spaces import Box
import numpy as np
from torchvision import transforms as T
import torch

from utils.agent import Agent


class PreprocessObservation(gym.ObservationWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.observation_space = Box(
            low=0,
            high=1,
            shape=(84, 84),
            dtype=np.float32,
        )

        def squeeze_tensor(x: torch.Tensor):
            return x.squeeze(0)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((84, 84)),
            T.Grayscale(),
            T.ToTensor(),
            T.Lambda(squeeze_tensor)
        ])

    def observation(self, observation):
        observation = self.transform(observation)
        assert observation.shape == self.observation_space.shape
        return observation


class ToTensorEnv(gym.ObservationWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)

        (h, w, c) = env.observation_space.shape

        self.observation_space = Box(low=0,
                                     high=1,
                                     dtype=np.float32,
                                     shape=(c, h, w))

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


O = TypeVar("O")
S = TypeVar("S")


def train(agent: Agent[O, S], training_frames=int(1e6)) -> Agent[O, S]:

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


def eval(agent: Agent[O, S], repeats=10) -> Agent[O, S]:
    agent.toggleEval(True)

    for _ in range(repeats):
        agent.reset()

        while True:

            (_, s) = agent.step()

            if s:
                break

    return agent


def train_and_eval(
        agent: Agent[O, S],
        single_train_frames=int(1e4),
        eval_repeats=10,
        total_train_frames=int(1e6),
) -> Agent[O, S]:

    for _ in tqdm(range(math.ceil(total_train_frames / single_train_frames))):
        train(agent, single_train_frames)
        eval(agent, eval_repeats)

    return agent
