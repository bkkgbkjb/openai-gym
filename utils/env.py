import gym
from typing import List, Tuple, Literal, Any, Optional, cast, Callable, Union, Iterable
from gym.wrappers import FrameStack
from gym.spaces import Box
import numpy as np
from torchvision import transforms as T


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
