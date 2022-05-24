import gym
import math
from typing import (
    Callable,
    Optional,
    Tuple,
    List,
    Any,
    cast,
    TypeVar,
)
from tqdm import tqdm
from time import sleep
from gym.spaces import Box
import numpy as np
from torchvision import transforms as T
import torch

from utils.agent import Agent
from utils.common import AllowedState


class PreprocessObservation(gym.ObservationWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.observation_space = Box(
            low=0,
            high=1,
            shape=(84, 84),
            dtype=np.float32,
        )

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((84, 84)),
            T.Grayscale(),
            T.ToTensor(),
            T.Lambda(lambda x: x.squeeze(0))
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

            sleep(1/60)
            (_, rwd, stop, _) = env.step(env.action_space.sample())
            t += 1

            r += rwd

            if stop:
                print(f"rwd is: {r}, total steps: {t}")
                break


O = TypeVar("O")
S = TypeVar("S", bound=AllowedState)


def train(agent: Agent[O], training_frames=int(1e6)) -> Agent[O]:

    agent.reset()

    with tqdm(total=training_frames) as pbar:
        frames = 0
        while frames <= training_frames:
            agent.reset()
            (_, (_, _, a, _)) = agent.train()

            pbar.update(len(a))
            frames += len(a)

    return agent


def eval(agent: Agent[O], env: gym.Env, repeats=10) -> Agent[O]:

    for _ in range(repeats):
        agent.reset()

        agent.eval(env)

    return agent


def train_and_eval(
        agent: Agent[O],
        eval_env: gym.Env,
        single_train_frames=int(1e4),
        eval_repeats=10,
        total_train_frames=int(1e6),
) -> Agent[O]:

    for _ in tqdm(range(math.ceil(total_train_frames / single_train_frames))):
        train(agent, single_train_frames)
        eval(agent, eval_env, eval_repeats)

    return agent


def make_train_and_eval_env(env_name: str,
                            wrappers: List[Callable[[gym.Env], gym.Env]] = [],
                            seed: int = 0) -> Tuple[gym.Env, gym.Env]:
    train_env = gym.make(env_name)
    train_env.seed(seed)
    train_env.action_space.seed(seed)
    train_env.observation_space.seed(seed)
    train_env.reset()

    # %%
    for w in wrappers:
        train_env = w(train_env)

    eval_env = gym.make(env_name)
    eval_env.seed(seed + 5)
    eval_env.action_space.seed(seed + 5)
    eval_env.observation_space.seed(seed + 5)
    eval_env.reset()

    # %%
    for w in wrappers:
        eval_env = w(eval_env)

    return train_env, eval_env
