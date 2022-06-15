import gym
import math
from typing import (
    Callable,
    Literal,
    Optional,
    Tuple,
    List,
    Union,
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

from utils.agent import Agent, OfflineAgent
from utils.agent import AllAgent
from utils.common import Action
from utils.env_sb3 import LazyFrames, resolve_lazy_frames

O = TypeVar('O')
S = TypeVar('S', bound=Union[torch.Tensor, LazyFrames])


def glance(env: gym.Env[O, Action],
           render: Union[Literal['rgb_array'], Literal['none'],
                         Literal['human']] = 'human',
           random_seed=0,
           repeats=3) -> gym.Env[O, Action]:
    assert render in ['rgb_array', 'none', 'human']
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
            if render != 'none':
                env.render(mode=render)
                sleep(1 / 60)

            (_, rwd, stop, _) = env.step(env.action_space.sample())
            t += 1

            r += rwd

            if stop:
                print(f"rwd is: {r}, total steps: {t}")
                break
    return env


def train(agent: Agent[O, S], training_frames=int(1e6)) -> Agent[O, S]:

    agent.reset()

    with tqdm(total=training_frames) as pbar:
        frames = 0
        while frames <= training_frames:
            agent.reset()
            (_, (_, _, a, _, _)) = agent.train()

            pbar.update(len(a))
            frames += len(a)

    return agent


def offline_train(
    agent: OfflineAgent[O, S],
    single_train_frames=int(1e6)) -> OfflineAgent[O, S]:

    agent.reset()

    with tqdm(total=single_train_frames) as pbar:
        frames = 0

        while frames <= single_train_frames:
            agent.reset()
            l = agent.train()

            pbar.update(l)
            frames += l

    return agent


def eval(agent: AllAgent[O, S],
         env: gym.Env[O, Action],
         repeats=10) -> AllAgent[O, S]:

    for _ in range(repeats):
        agent.reset()

        agent.eval(env)

    return agent


def train_and_eval(
        agent: Agent[O, S],
        eval_env: gym.Env[O, Action],
        single_train_frames=int(1e4),
        eval_repeats=10,
        total_train_frames=int(1e6),
) -> Agent[O, S]:
    s = math.ceil(total_train_frames / single_train_frames)

    for _ in tqdm(range(s)):
        train(agent, single_train_frames)
        eval(agent, eval_env, eval_repeats)

    return agent


def offline_train_and_eval(
    agent: OfflineAgent[O, S],
    eval_env: gym.Env[O, Action],
    single_train_frames=int(1e4),
    eval_repeats=10,
    total_train_frames=int(1e6)) -> OfflineAgent[O, S]:

    s = math.ceil(total_train_frames / single_train_frames)

    for _ in tqdm(range(s)):
        offline_train(agent, single_train_frames)
        eval(agent, eval_env, eval_repeats)

    return agent


def make_train_and_eval_env(
        env_name: str,
        wrappers: List[
            Callable[[gym.Env, Union[Literal['train'], Literal['eval']]],
                     gym.Env]] = [],
        seed: int = 0) -> Tuple[gym.Env[O, Action], gym.Env[O, Action]]:
    train_env = gym.make(env_name)
    train_env.seed(seed)
    train_env.action_space.seed(seed)
    train_env.observation_space.seed(seed)
    train_env.reset()

    # %%
    for w in wrappers:
        train_env = w(train_env, 'train')

    eval_env = gym.make(env_name)
    eval_env.seed(seed + 5)
    eval_env.action_space.seed(seed + 5)
    eval_env.observation_space.seed(seed + 5)
    eval_env.reset()

    # %%
    for w in wrappers:
        eval_env = w(eval_env, 'eval')

    return train_env, eval_env
