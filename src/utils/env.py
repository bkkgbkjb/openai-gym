import gym
from typing import List, Tuple, Literal, Any, Optional, cast, Callable, Union, Iterable
from gym.wrappers import FrameStack
from gym.spaces import Box
import numpy as np
from torchvision import transforms as T


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

        self.observation_space = Box(
            low=0, high=1, dtype=np.float32, shape=(c, h, w))

        self.transform = T.ToTensor()

    def observation(self, observation):
        observation = self.transform(observation)
        assert observation.shape == self.observation_space.shape
        return observation
