import gym
import math
from typing import (
    Callable,
    Literal,
    Dict,
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
from torch.utils.data import DataLoader

from utils.agent import Agent, OfflineAgent
from utils.agent import AllAgent
from utils.common import Action
from datetime import datetime
from utils.env_sb3 import LazyFrames, RecordVideo, resolve_lazy_frames

O = TypeVar('O')
S = TypeVar('S', bound=Union[torch.Tensor, LazyFrames])


def glance(env: gym.Env,
           render: Union[Literal['rgb_array'], Literal['none'],
                         Literal['human']] = 'human',
           random_seed=0,
           repeats=3) -> gym.Env:
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


def train(agent: Agent[O, S], train_env: gym.Env,
          training_frames=int(1e6)) -> Agent[O, S]:

    with tqdm(total=training_frames) as pbar:
        frames = 0
        while frames <= training_frames:
            (_, (_, _, a, _, _)) = agent.train(train_env)

            pbar.update(len(a))
            frames += len(a)

    return agent


def offline_train(
    agent: OfflineAgent[O, S],
    dataloader: DataLoader,
    single_train_frames=int(1e6)) -> OfflineAgent[O, S]:

    with tqdm(total=single_train_frames) as pbar:
        frames = 0

        while frames <= single_train_frames:
            l = agent.train(dataloader)

            pbar.update(l)
            frames += l

    return agent


def eval(agent: AllAgent[O, S],
         env: Union[gym.Env, List[gym.Env]],
         repeats=10,
         env_weights: Optional[List[float]] = None) -> AllAgent[O, S]:

    report_dict: Dict[str, Tuple[Any, int]] = dict()

    for _ in tqdm(range(repeats)):

        (report_info,
         _) = agent.eval(env if not isinstance(env, list) else env[
             np.random.choice(len(env), p=env_weights)])

        for (k, v) in report_info.items():

            if k not in report_dict:
                report_dict[k] = (v, 1)
            else:
                old_v = report_dict[k][0]
                n = report_dict[k][1] + 1

                report_dict[k] = (old_v + (v - old_v) / n, n)

    for (k, (v, _)) in report_dict.items():
        agent.report({f'eval/{k}': v})

    return agent


def train_and_eval(
        agent: Agent[O, S],
        train_env: gym.Env,
        eval_env: Union[gym.Env, List[gym.Env]],
        single_train_frames=int(1e4),
        eval_repeats=10,
        total_train_frames=int(1e6),
        eval_per_train=1,
        eval_env_weights: Optional[List[float]] = None) -> Agent[O, S]:
    s = math.ceil(total_train_frames / single_train_frames)

    t = 0

    eval(agent, eval_env, eval_repeats, env_weights=eval_env_weights)
    for _ in tqdm(range(s)):
        train(agent, train_env, single_train_frames)
        t += 1
        if t == eval_per_train:
            eval(agent, eval_env, eval_repeats, env_weights=eval_env_weights)
            t = 0

    return agent


def offline_train_and_eval(
        agent: OfflineAgent[O, S],
        dataloader: DataLoader,
        eval_env: Union[gym.Env, List[gym.Env]],
        single_train_frames=int(1e4),
        eval_repeats=10,
        total_train_frames=int(1e6),
        eval_per_train=1,
        eval_env_weights: Optional[List[float]] = None) -> OfflineAgent[O, S]:

    s = math.ceil(total_train_frames / single_train_frames)
    t = 0

    eval(agent, eval_env, eval_repeats, env_weights=eval_env_weights)
    for _ in tqdm(range(s)):
        offline_train(agent, dataloader, single_train_frames)
        t += 1
        if t == eval_per_train:
            eval(agent, eval_env, eval_repeats, env_weights=eval_env_weights)
            t = 0

    return agent


def make_train_and_eval_env(envs: Union[str, Tuple[gym.Env, gym.Env]],
                            wrappers: List[Callable[[gym.Env], gym.Env]] = [],
                            seed: int = 0) -> Tuple[gym.Env, gym.Env]:
    train_env = gym.make(envs) if isinstance(envs, str) else envs[0]
    train_env.seed(seed)
    train_env.action_space.seed(seed)
    train_env.observation_space.seed(seed)
    train_env.reset()

    # %%
    for w in wrappers:
        train_env = w(train_env)

    eval_env = gym.make(envs) if isinstance(envs, str) else envs[1]
    eval_env.seed(seed + 5)
    eval_env.action_space.seed(seed + 5)
    eval_env.observation_space.seed(seed + 5)
    eval_env.reset()

    # %%
    for w in wrappers:
        eval_env = w(eval_env)

    return train_env, eval_env


def make_envs(
        envs: Union[Tuple[str, int], List[gym.Env]],
        wrappers: List[Callable[[gym.Env], gym.Env]] = []) -> List[gym.Env]:
    es = []
    if isinstance(envs, tuple):
        for _ in range(envs[1]):
            es.append(gym.make(envs[0]))
    else:
        es = envs

    for w in wrappers:
        es = list(map(w, es))

    seed = 0
    for i, e in enumerate(es):
        e.seed(seed + i * 5)

    return es


def record_video(env: gym.Env,
                 algo_name: str,
                 activate_per_episode: int = 1,
                 name_prefix: str = '') -> gym.Env:
    return RecordVideo(
        env,
        f'vlog/{algo_name}_{datetime.now().strftime("%m-%d_%H-%M")}',
        episode_trigger=lambda episode_id: episode_id % activate_per_episode ==
        0,
        name_prefix=
        f'{algo_name}{f"_{name_prefix}" if name_prefix != "" else ""}')
